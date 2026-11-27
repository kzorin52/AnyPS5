#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <initializer_list>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

size_t BlockAt(const Program& program, const Block* block) {
	return static_cast<size_t>(std::ranges::find(program.blocks, block) - program.blocks.begin());
}

bool LeaveAfter(Block& exit, Block& stay, Value done, uint32_t limit, bool mark) {
	auto& term = exit.terminator;
	if (exit.condition.IsEmpty() || term.kind != CFG::TerminatorKind::ConditionalBranch) return false;
	const bool stay_on_true = term.true_block == &stay;
	if (!stay_on_true && term.false_block != &stay) return false;
	const auto add = [&](ValueOpcode op, std::initializer_list<Value> args) {
		return Value(&*exit.PrependNewInst(exit.end(), op, args));
	};
	const auto under = add(ValueOpcode::ULessThan32, {done, Value(limit)});
	Value      over;
	if (mark || !stay_on_true) over = add(ValueOpcode::LogicalNot, {under});
	if (mark) {
		const auto stays = stay_on_true ? exit.condition : add(ValueOpcode::LogicalNot, {exit.condition});
		add(ValueOpcode::LoopGuardMark, {add(ValueOpcode::LogicalAnd, {stays, over})});
	}
	exit.condition = stay_on_true ? add(ValueOpcode::LogicalAnd, {exit.condition, under})
	                              : add(ValueOpcode::LogicalOr, {exit.condition, over});
	return true;
}

bool ZeroAvailable(Type type) {
	switch (type) {
		case Type::U1:
		case Type::U8:
		case Type::U16:
		case Type::U32:
		case Type::U64:
		case Type::F16:
		case Type::F32:
		case Type::F64:
		case Type::U32x2:
		case Type::U32x3:
		case Type::U32x4:
		case Type::F32x2: return true;
		default: return false;
	}
}

Value ZeroOf(Block& block, Type type) {
	const auto at = [&](ValueOpcode op, std::initializer_list<Value> args) {
		return Value(&*block.PrependNewInst(block.end(), op, args));
	};
	switch (type) {
		case Type::U1: return Value(false);
		case Type::U8: return Value(uint8_t {0});
		case Type::U16: return Value(uint16_t {0});
		case Type::U32: return Value(0u);
		case Type::U64: return Value(uint64_t {0});
		case Type::F16: return at(ValueOpcode::BitCastF16U16, {Value(uint16_t {0})});
		case Type::F32: return at(ValueOpcode::BitCastF32U32, {Value(0u)});
		case Type::F64: return at(ValueOpcode::BitCastF64U64, {Value(uint64_t {0})});
		case Type::U32x2: return at(ValueOpcode::CompositeConstructU32x2, {Value(0u), Value(0u)});
		case Type::U32x3: return at(ValueOpcode::CompositeConstructU32x3, {Value(0u), Value(0u), Value(0u)});
		case Type::U32x4: return at(ValueOpcode::CompositeConstructU32x4, {Value(0u), Value(0u), Value(0u), Value(0u)});
		case Type::F32x2: {
			const auto zero = at(ValueOpcode::BitCastF32U32, {Value(0u)});
			return at(ValueOpcode::CompositeConstructF32x2, {zero, zero});
		}
		default: return {};
	}
}

Block* EveryIterationExit(Program& program, Block& header, Block* latch, size_t first, size_t last, Block*& stay) {
	auto* merge = header.terminator.merge_block;
	auto* at    = &header;
	for (size_t steps = 0; at != nullptr && at != latch && steps <= last - first; ++steps) {
		const auto index = BlockAt(program, at);
		if (index < first || index > last) return nullptr;
		const auto& term = at->terminator;
		if (term.kind == CFG::TerminatorKind::Branch) {
			at = at != &header && term.loop_header && term.merge_block != nullptr ? term.merge_block : term.true_block;
			continue;
		}
		if (term.kind != CFG::TerminatorKind::ConditionalBranch || at->condition.IsEmpty()) return nullptr;
		if (term.true_block != merge && term.false_block != merge) {
			if (term.merge_block == nullptr || (!term.loop_header && BlockAt(program, term.merge_block) <= index)) return nullptr;
			at = term.merge_block;
			continue;
		}
		stay = term.true_block == merge ? term.false_block : term.true_block;
		return stay != merge ? at : nullptr;
	}
	return nullptr;
}

bool InsideLoop(const Program& program, const Block* block, size_t first, size_t last) {
	const auto at = BlockAt(program, block);
	return at >= first && at <= last;
}

bool BodyValuesLeave(const Program& program, size_t first, size_t last, const Block* merge) {
	for (auto at = first + 1; at <= last; ++at) {
		for (const auto& inst: *program.blocks[at]) {
			for (const auto& use: inst.Uses()) {
				const auto* block = use.user->Parent();
				if (InsideLoop(program, block, first, last)) continue;
				if (block == merge && use.user->GetOpcode() == ValueOpcode::Phi) continue;
				return true;
			}
		}
	}
	for (size_t at = 0; at < program.blocks.size(); ++at) {
		if (at >= first && at <= last) continue;
		for (const auto value: {program.blocks[at]->condition, program.blocks[at]->indirect_target}) {
			const auto* inst = value.IsEmpty() ? nullptr : value.Resolve().TryInstruction();
			if (inst != nullptr && inst->Parent() != nullptr && InsideLoop(program, inst->Parent(), first + 1, last)) return true;
		}
	}
	return false;
}

bool DominatesLatch(const Program& program, const Block* block, const Block& header, const Block* latch, size_t first,
                    size_t last) {
	if (block == &header || block == latch) return true;
	std::vector<bool>         seen(program.blocks.size());
	std::vector<const Block*> stack {&header};
	seen[first] = true;
	while (!stack.empty()) {
		const auto* at = stack.back();
		stack.pop_back();
		for (const auto* next: at->ImmSuccessors()) {
			if (next == block || !InsideLoop(program, next, first, last)) continue;
			if (next == latch) return false;
			const auto index = BlockAt(program, next);
			if (seen[index]) continue;
			seen[index] = true;
			stack.push_back(next);
		}
	}
	return true;
}

Value ExitValue(const Program& program, const Inst& phi, const Block& header, const Block* latch, size_t first, size_t last) {
	Value common;
	for (size_t index = 0; index < phi.NumArgs(); ++index) {
		if (!InsideLoop(program, phi.PhiBlock(index), first, last)) return {};
		const auto value = phi.Arg(index).Resolve();
		if (index != 0 && !(value == common)) return {};
		common = value;
	}
	if (common.IsEmpty() || common.IsImmediate()) return common;
	const auto* inst = common.TryInstruction();
	if (inst == nullptr || inst->Parent() == nullptr) return {};
	if (!InsideLoop(program, inst->Parent(), first, last)) return common;
	return DominatesLatch(program, inst->Parent(), header, latch, first, last) ? common : Value {};
}

bool HasBarrier(const Block& block) {
	return std::ranges::any_of(block, [](const Inst& inst) { return inst.GetOpcode() == ValueOpcode::Barrier; });
}

bool ReachesBarrier(const Program& program, const Block& header) {
	std::vector<bool>         seen(program.blocks.size());
	std::vector<const Block*> stack {&header};
	seen[BlockAt(program, &header)] = true;
	while (!stack.empty()) {
		const auto* at = stack.back();
		stack.pop_back();
		if (HasBarrier(*at)) return true;
		for (const auto* next: at->ImmSuccessors()) {
			const auto index = BlockAt(program, next);
			if (index >= seen.size() || seen[index]) continue;
			seen[index] = true;
			stack.push_back(next);
		}
	}
	return false;
}

using Op = ValueOpcode;

enum class BranchScope { Unknown, Constant, WaveVote };

BranchScope ScopeOfBranch(Value condition) {
	for (unsigned depth = 0; depth < 24; ++depth) {
		condition = condition.Resolve();
		if (condition.IsImmediate()) return condition.GetType() == Type::U1 ? BranchScope::Constant : BranchScope::Unknown;
		const auto* inst = condition.Instruction();
		if (inst->GetOpcode() == Op::LogicalNot) {
			condition = inst->Arg(0);
			continue;
		}
		if (inst->GetOpcode() != Op::ConditionRef) return BranchScope::Unknown;
		switch (inst->Flags<CFG::BranchCondition>()) {
			case CFG::BranchCondition::ExecZero:
			case CFG::BranchCondition::ExecNonZero:
			case CFG::BranchCondition::VccZero:
			case CFG::BranchCondition::VccNonZero:
			case CFG::BranchCondition::SccZero:
			case CFG::BranchCondition::SccNonZero: return BranchScope::WaveVote;
			case CFG::BranchCondition::ScalarInstruction:
				condition = inst->Arg(0);
				break;
			default: return BranchScope::Unknown;
		}
	}
	return BranchScope::Unknown;
}

bool SupportsFixedTails(const Program& program) {
	return std::ranges::all_of(program.blocks, [](const Block* block) {
		if (std::ranges::any_of(*block, [](const Inst& inst) {
			return inst.GetOpcode() == Op::UndefU1 || inst.GetOpcode() == Op::UndefU32;
		})) return false;
		switch (block->terminator.kind) {
			case CFG::TerminatorKind::Branch:
			case CFG::TerminatorKind::Return: return true;
			case CFG::TerminatorKind::ConditionalBranch: return ScopeOfBranch(block->condition) != BranchScope::Unknown;
			default: return false;
		}
	});
}

struct FixedTraversal {
	Inst* cursor {};
	Inst* step {};
	size_t back {};
	Value records;
	Value active;
};

bool PlainLoopInstruction(Op op) {
	switch (op) {
		case Op::Phi:
		case Op::Reference:
		case Op::ConditionRef:
		case Op::SelectU32:
		case Op::LogicalNot:
		case Op::LogicalAnd:
		case Op::LogicalOr:
		case Op::IAdd32:
		case Op::ISub32:
		case Op::BitwiseAnd32:
		case Op::BitwiseOr32:
		case Op::ShiftLeftLogical32:
		case Op::ShiftRightLogical32:
		case Op::IEqual32:
		case Op::INotEqual32:
		case Op::SGreaterThan32:
		case Op::UMin32:
		case Op::BitCastF32U32:
		case Op::BitCastU32F32:
		case Op::FPOrdEqual32:
		case Op::FPMin32:
		case Op::FPMax32:
		case Op::FPMinTri32:
		case Op::FPMaxTri32:
		case Op::CompositeExtractU32x2:
		case Op::CompositeExtractU32x3:
		case Op::CompositeExtractU32x4:
		case Op::GetBufferResource: return true;
		default: return false;
	}
}

bool PlainRead(const Program& program, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != Op::LoadBufferU32 && op != Op::LoadBufferU32x2 && op != Op::LoadBufferU32x3 &&
	    op != Op::LoadBufferU32x4) return false;
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= program.memory_info.size()) return false;
	const auto& memory = program.memory_info[index];
	return memory.kind == ResourceKind::IndirectBuffer && !memory.coherent && !memory.typed &&
	       !memory.formatted && memory.data_bits == 32u;
}

Value IndexedReadBound(const Program& program, const Inst& read, size_t first, size_t last) {
	const auto& memory = program.memory_info[read.Flags<MemoryFlags>().index];
	if (!memory.idxen || read.Arg(3) != Value(0u)) return {};
	const auto* desc = read.Arg(0).Resolve().TryInstruction();
	if (desc == nullptr || desc->GetOpcode() != Op::GetBufferResource) return {};
	const auto format = desc->Arg(3).Resolve();
	if (!format.IsImmediate() || format.GetType() != Type::U32) return {};
	const auto oob_mode = (format.U32() >> 28u) & 3u;
	const bool add_tid = ((format.U32() >> 23u) & 1u) != 0;
	if (oob_mode >= 2 || add_tid) return {};
	const auto bound = desc->Arg(2).Resolve();
	const auto* origin = bound.TryInstruction();
	return origin != nullptr && InsideLoop(program, origin->Parent(), first, last) ? Value {} : bound;
}

bool FixedCursor(const Program& program, Inst& phi, size_t back, size_t first, size_t last, FixedTraversal& plan) {
	auto* step = phi.Arg(back).Resolve().TryInstruction();
	if (step == nullptr || step->GetOpcode() != Op::SelectU32 || step->Arg(2) != Value(&phi)) return false;
	auto* add = step->Arg(1).Resolve().TryInstruction();
	if (add == nullptr || add->GetOpcode() != Op::IAdd32) return false;
	const auto one = Value(1u);
	if (!((add->Arg(0) == one && add->Arg(1) == Value(&phi)) ||
	      (add->Arg(1) == one && add->Arg(0) == Value(&phi)))) return false;
	if (add->UseCount() != 1) return false;
	Value records;
	for (const auto& use: phi.Uses()) {
		const auto* user = use.user;
		if (!InsideLoop(program, user->Parent(), first, last) || user == step || user == add) continue;
		if (!PlainRead(program, *user) || use.operand != 1 || user->Arg(4) != step->Arg(0)) return false;
		const auto bound = IndexedReadBound(program, *user, first, last);
		if (bound.IsEmpty()) return false;
		if (!records.IsEmpty() && records != bound) return false;
		records = bound;
	}
	if (records.IsEmpty() || step->Parent() != program.blocks[last]) return false;
	if (std::ranges::any_of(step->Uses(), [&](const Use& use) { return use.user != &phi; })) return false;
	plan = {&phi, step, back, records, step->Arg(0)};
	return true;
}

FixedTraversal FindFixedTraversal(const Program& program, size_t first, size_t last) {
	FixedTraversal plan;
	auto* header = program.blocks[first];
	auto* latch = program.blocks[last];
	if (header->ImmPredecessors().size() != 2 || latch == header) return {};
	for (size_t at = first; at <= last; ++at) {
		auto& block = *program.blocks[at];
		if (at != first && (block.terminator.loop_header || block.ImmPredecessors().empty())) return {};
		if (block.terminator.kind != CFG::TerminatorKind::Branch &&
		    block.terminator.kind != CFG::TerminatorKind::ConditionalBranch) return {};
		if (at != first) {
			for (const auto* pred: block.ImmPredecessors()) {
				if (!InsideLoop(program, pred, first, at - 1)) return {};
			}
		}
		for (auto& inst: block) {
			if (!PlainLoopInstruction(inst.GetOpcode()) && !PlainRead(program, inst)) return {};
			if (at == first && inst.GetOpcode() == Op::Phi) {
				if ((inst.GetType() != Type::U32 && inst.GetType() != Type::U1) || inst.NumArgs() != 2) return {};
				const size_t back = inst.PhiBlock(0) == latch ? 0 : 1;
				if (inst.PhiBlock(back) != latch) return {};
				if (inst.GetType() == Type::U32) {
					FixedTraversal found;
					if (FixedCursor(program, inst, back, first, last, found)) {
						if (plan.cursor != nullptr) return {};
						plan = found;
					}
				}
			}
		}
	}
	return plan;
}

bool InertWhenClear(const Program& program, size_t first, size_t last, Inst& mask) {
	std::unordered_map<const Inst*, Value> known;
	const auto get = [&](Value arg) {
		arg = arg.Resolve();
		const auto* inst = arg.TryInstruction();
		if (inst == nullptr || !InsideLoop(program, inst->Parent(), first, last)) return arg;
		const auto it = known.find(inst);
		return it == known.end() ? Value {} : it->second;
	};
	const Value on(true), off(false);
	auto& header = *program.blocks[first];
	const auto* latch = program.blocks[last];
	for (size_t at = first; at <= last; ++at) {
		for (auto& inst: *program.blocks[at]) {
			const auto op = inst.GetOpcode();
			Value result;
			if (op == Op::Phi && at == first) {
				result = &inst == &mask ? off : Value(&inst);
			} else if (op == Op::Phi) {
				result = get(inst.Arg(0));
				for (size_t n = 1; n < inst.NumArgs(); ++n) {
					if (result != get(inst.Arg(n))) { result = {}; break; }
				}
			} else if (op == Op::SelectU32) {
				const auto condition = get(inst.Arg(0));
				const auto yes = get(inst.Arg(1));
				const auto no = get(inst.Arg(2));
				if (condition == on || yes == no) result = yes;
				else if (condition == off) result = no;
			} else if (op == Op::LogicalNot) {
				const auto value = get(inst.Arg(0));
				if (value == on) result = off;
				else if (value == off) result = on;
			} else if (op == Op::LogicalAnd || op == Op::LogicalOr) {
				const auto left = get(inst.Arg(0)), right = get(inst.Arg(1));
				const auto absorbing = op == Op::LogicalAnd ? off : on;
				const auto neutral = op == Op::LogicalAnd ? on : off;
				if (left == absorbing || right == absorbing) result = absorbing;
				else if (left == neutral || left == right) result = right;
				else if (right == neutral) result = left;
			} else if (op == Op::Reference) {
				if (inst.Arg(0) != program.blocks[at]->condition) return false;
			} else if (op == Op::ConditionRef) {
				const auto value = get(inst.Arg(0));
				if (value != off && value != on) return false;
			} else if (PlainRead(program, inst)) {
				if (get(inst.Arg(4)) != off) return false;
			} else if (!PlainLoopInstruction(op)) {
				return false;
			}
			if (result.IsEmpty()) {
				if (std::ranges::any_of(inst.Uses(), [&](const Use& use) {
					return !InsideLoop(program, use.user->Parent(), first, last);
				})) return false;
			} else {
				known.emplace(&inst, result);
			}
		}
	}
	for (auto& phi: header) {
		if (phi.GetOpcode() != Op::Phi) break;
		const auto back = phi.PhiBlock(0) == latch ? 0 : 1;
		if (get(phi.Arg(back)) != get(Value(&phi))) return false;
	}
	return true;
}

Value Append(Block& block, Op op, std::initializer_list<Value> args) {
	return Value(&block.AppendNewInst(op, args));
}

Value DeterminateInputs(const Program& program, size_t first, size_t last) {
	std::vector<Value> clean(last - first + 1);
	for (size_t at = first; at <= last; ++at) {
		auto& block = *program.blocks[at];
		Value valid(true);
		if (at != first) {
			const auto preds = block.ImmPredecessors();
			valid = clean[BlockAt(program, preds.front()) - first];
			if (!std::ranges::all_of(preds, [&](auto* pred) { return clean[BlockAt(program, pred) - first] == valid; })) {
				auto& phi = *block.PrependNewInst(block.begin(), Op::Phi);
				phi.SetFlags(Type::U1);
				for (auto* pred: preds) phi.AddPhiOperand(pred, clean[BlockAt(program, pred) - first]);
				valid = Value(&phi);
			}
		}
		std::vector<Value> float_inputs;
		std::vector<Value> shifts;
		for (const auto& inst: block) {
			const auto op = inst.GetOpcode();
			if (op == Op::Phi) continue;
			if (op == Op::ShiftLeftLogical32 || op == Op::ShiftRightLogical32) shifts.push_back(inst.Arg(1));
			for (size_t i = 0; i < inst.NumArgs(); ++i) {
				const auto arg = inst.Arg(i).Resolve();
				if (arg.GetType() != Type::F32 || std::ranges::find(float_inputs, arg) != float_inputs.end()) continue;
				float_inputs.push_back(arg);
			}
		}
		for (const auto arg: float_inputs) {
			const auto* producer = arg.TryInstruction();
			const auto bits = producer != nullptr && producer->GetOpcode() == Op::BitCastF32U32
			    ? producer->Arg(0) : Append(block, Op::BitCastU32F32, {arg});
			const auto magnitude = Append(block, Op::BitwiseAnd32, {bits, Value(0x7fffffffu)});
			const auto non_nan = Append(block, Op::ULessThanEqual32, {magnitude, Value(0x7f800000u)});
			valid = Append(block, Op::LogicalAnd, {valid, non_nan});
		}
		for (const auto shift: shifts) {
			if (shift.IsImmediate() && shift.U32() < 32u) continue;
			valid = Append(block, Op::LogicalAnd, {valid, Append(block, Op::ULessThan32, {shift, Value(32u)})});
		}
		clean[at - first] = valid;
	}
	return clean.back();
}

void SkipFixedTail(const Program& program, const FixedTraversal& plan, Inst& count, Value next, uint32_t limit,
                   size_t first, size_t last, Value inert_mask) {
	auto& latch = *program.blocks[last];
	const auto add = [&](Op op, std::initializer_list<Value> args) { return Append(latch, op, args); };
	auto stable = DeterminateInputs(program, first, last);
	if (!inert_mask.IsEmpty()) stable = add(Op::LogicalOr, {add(Op::LogicalNot, {inert_mask}), stable});
	for (auto& phi: *program.blocks[first]) {
		if (phi.GetOpcode() != Op::Phi) break;
		if (&phi == &count || &phi == plan.cursor) continue;
		const auto value = phi.Arg(phi.PhiBlock(0) == &latch ? 0 : 1);
		const auto same = phi.GetType() == Type::U32 ? add(Op::IEqual32, {Value(&phi), value})
		    : add(Op::LogicalNot, {add(Op::LogicalXor, {Value(&phi), value})});
		stable = add(Op::LogicalAnd, {stable, same});
	}
	const auto remaining = add(Op::ISub32, {Value(limit), next});
	const auto cursor = Value(plan.cursor);
	const auto tail = add(Op::UGreaterThanEqual32, {cursor, plan.records});
	const auto room = add(Op::ISub32, {Value(UINT32_MAX), Value(plan.step)});
	const auto no_wrap = add(Op::ULessThanEqual32, {remaining, room});
	const auto monotone = add(Op::UGreaterThanEqual32, {Value(plan.step), cursor});
	const auto advances = add(Op::LogicalAnd, {tail, add(Op::LogicalAnd, {no_wrap, monotone})});
	const auto safe = add(Op::LogicalOr, {add(Op::LogicalNot, {plan.active}), advances});
	const auto good = add(Op::LogicalAnd, {stable, safe});
	auto& header = *program.blocks[first];
	auto& previous = *header.PrependNewInst(header.begin(), Op::Phi);
	previous.SetFlags(Type::U1);
	for (auto* pred: header.ImmPredecessors()) previous.AddPhiOperand(pred, pred == &latch ? good : Value(false));
	const auto changed = add(Op::LogicalNot, {add(Op::LogicalAnd, {Value(&previous), good})});
	const Value any(&latch.AppendNewInst(Op::ConditionRef, {changed},
	                                    static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero)));
	const auto skip = add(Op::LogicalNot, {any});
	const auto delta = add(Op::SelectU32, {add(Op::LogicalAnd, {skip, plan.active}), remaining, Value(0u)});
	plan.cursor->SetArg(plan.back, add(Op::IAdd32, {Value(plan.step), delta}));
	count.SetArg(count.PhiBlock(0) == &latch ? 0 : 1, add(Op::SelectU32, {skip, Value(limit), next}));
}

}

static const char* GuardLoopImpl(Program& program, Block& header, uint32_t limit, LoopExit mode, bool fixed_tail) {
	const auto first = BlockAt(program, &header);
	const auto last  = BlockAt(program, header.terminator.continue_block);
	auto*      merge = header.terminator.merge_block;
	Block*     latch = nullptr;
	for (auto* pred: header.ImmPredecessors()) {
		const auto at = BlockAt(program, pred);
		if (at < first || at > last) continue;
		if (latch != nullptr) return "several back edges";
		latch = pred;
	}
	if (latch == nullptr || merge == nullptr || merge == &header) return "no back edge or no merge block";
	auto&      term        = latch->terminator;
	const bool conditional = term.kind == CFG::TerminatorKind::ConditionalBranch && !latch->condition.IsEmpty() &&
	                         (term.true_block == &header || term.false_block == &header);
	const bool plain       = term.kind == CFG::TerminatorKind::Branch && term.true_block == &header &&
	                         std::ranges::find(latch->ImmSuccessors(), merge) == latch->ImmSuccessors().end();
	if (!conditional && !plain) return "back edge is neither conditional nor plain";
	const bool mark      = mode == LoopExit::Guard;
	const auto zero_exit = [&](const Inst& phi) { return mode == LoopExit::EmptyWalk || !phi.HasUses(); };
	Block*     exit      = nullptr;
	Block*     stay      = nullptr;
	if (plain) {
		exit = EveryIterationExit(program, header, latch, first, last, stay);
		if (exit == nullptr) {
			if (BodyValuesLeave(program, first, last, merge)) return "a value of the loop body is used after the loop";
			for (const auto& inst: *merge) {
				if (inst.GetOpcode() != ValueOpcode::Phi) break;
				if (zero_exit(inst) ? !ZeroAvailable(inst.GetType())
				                    : ExitValue(program, inst, header, latch, first, last).IsEmpty()) {
					return "a merge value has no known value on a new exit edge";
				}
			}
		}
	}
	const bool can_skip = fixed_tail && plain && exit != nullptr && latch == program.blocks[last] &&
	                      ScopeOfBranch(exit->condition) == BranchScope::WaveVote;
	const auto plan = can_skip ? FindFixedTraversal(program, first, last) : FixedTraversal {};
	Value inert_mask;
	if (plan.cursor != nullptr) {
		for (auto& phi: header) {
			if (phi.GetOpcode() != Op::Phi) break;
			if (phi.GetType() == Type::U1 && InertWhenClear(program, first, last, phi)) {
				inert_mask = Value(&phi);
				break;
			}
		}
	}
	auto& count = *header.PrependNewInst(header.begin(), ValueOpcode::Phi);
	count.SetFlags(Type::U32);
	const auto body = std::ranges::find_if(header, [](const Inst& inst) { return inst.GetOpcode() != ValueOpcode::Phi; });
	auto&      next = *header.PrependNewInst(body, ValueOpcode::IAdd32, {Value(&count), Value(1u)});
	for (auto* pred: header.ImmPredecessors()) {
		count.AddPhiOperand(pred, pred == latch ? Value(&next) : Value(0u));
	}
	if (plan.cursor != nullptr) SkipFixedTail(program, plan, count, Value(&next), limit, first, last, inert_mask);
	if (conditional) {
		LeaveAfter(*latch, header, Value(&next), limit, mark);
		return nullptr;
	}
	if (exit != nullptr) {
		LeaveAfter(*exit, *stay, Value(&count), limit, mark);
		return nullptr;
	}
	for (auto& inst: *merge) {
		if (inst.GetOpcode() != ValueOpcode::Phi) break;
		inst.AddPhiOperand(latch, zero_exit(inst) ? ZeroOf(*latch, inst.GetType())
		                                          : ExitValue(program, inst, header, latch, first, last));
	}
	auto& under = *latch->PrependNewInst(latch->end(), ValueOpcode::ULessThan32, {Value(&next), Value(limit)});
	if (mark) {
		latch->PrependNewInst(latch->end(), ValueOpcode::LoopGuardMark,
		                      {Value(&*latch->PrependNewInst(latch->end(), ValueOpcode::LogicalNot, {Value(&under)}))});
	}
	latch->condition = Value(&under);
	term.kind        = CFG::TerminatorKind::ConditionalBranch;
	term.true_block  = &header;
	term.false_block = merge;
	latch->AddBranch(merge);
	return nullptr;
}

const char* GuardLoop(Program& program, Block& header, uint32_t limit, LoopExit mode) {
	return GuardLoopImpl(program, header, limit, mode, false);
}

LoopGuards GuardComputeLoops(Program& program, uint32_t limit, bool several_waves, std::span<Block* const> bounded) {
	LoopGuards guards;
	if (program.dispatcher_fallback) {
		guards.unguarded = 1;
		return guards;
	}
	const bool rendezvous =
	    several_waves && std::ranges::any_of(program.blocks, [](const Block* block) { return HasBarrier(*block); });
	std::vector<Block*> headers;
	for (auto* block: program.blocks) {
		const auto& term = block->terminator;
		if (term.loop_header && term.continue_block != nullptr && term.merge_block != nullptr) headers.push_back(block);
	}
	const bool fixed_tail = !headers.empty() && !several_waves && SupportsFixedTails(program);
	for (auto* header: headers) {
		if (std::ranges::find(bounded, header) != bounded.end()) {
			++guards.bounded;
			continue;
		}
		const char* refusal = rendezvous && ReachesBarrier(program, *header)
		                          ? "a barrier shared by several waves can follow the loop exit"
		                          : GuardLoopImpl(program, *header, limit, LoopExit::Guard, fixed_tail);
		if (refusal == nullptr) {
			++guards.guarded;
			continue;
		}
		++guards.unguarded;
		guards.refusals.emplace_back(header->id, refusal);
	}
	return guards;
}

}
