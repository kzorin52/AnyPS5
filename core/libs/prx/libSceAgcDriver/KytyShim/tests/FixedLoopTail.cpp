#include "graphics/shader/recompiler/ir/passes/LoopGuard.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <stdexcept>
#include <map>
#include <memory>

namespace {

using namespace Libs::Graphics::ShaderRecompiler;
using namespace IR;
using Op = ValueOpcode;
using Wave = std::array<uint32_t, 64>;

enum class Variant { Plain, Coherent, Atomic, Barrier, TwoWaves, OuterLane, ScalarLane, OtherMask, EscapeStep, NoBallot, Dormant, DormantRead, DormantVote, DormantEscapes, Revived };

struct Inputs {
	Wave start;
	Wave active;
	Wave data {};
};

struct Result {
	Wave cursor;
	Wave payload;
	uint32_t marks;
	uint32_t turns;
	bool rewritten;
	bool guarded;
};

Result Run(const Inputs& input, Variant variant) {
	constexpr uint32_t limit = 16;
	Program program;
	for (uint32_t id = 0; id < 5; ++id) {
		auto& block = *program.block_storage.emplace_back(std::make_unique<Block>());
		block.id = id;
		program.blocks.push_back(&block);
	}
	auto& entry = *program.blocks[0];
	auto& header = *program.blocks[1];
	auto& test = *program.blocks[2];
	auto& latch = *program.blocks[3];
	auto& exit = *program.blocks[4];
	auto link = [](Block& from, Block& to) {
		from.terminator.kind = CFG::TerminatorKind::Branch;
		from.terminator.true_block = &to;
		from.AddBranch(&to);
	};
	link(entry, header);
	link(header, test);
	link(test, latch);
	link(latch, header);
	header.terminator.loop_header = true;
	header.terminator.merge_block = &exit;
	header.terminator.continue_block = &latch;
	test.terminator.kind = CFG::TerminatorKind::ConditionalBranch;
	test.terminator.false_block = &exit;
	test.AddBranch(&exit);

	const auto input_at = [&](uint32_t index) {
		return Value(&entry.AppendNewInst(Op::GetUserData, {Value(static_cast<ScalarReg>(index))}));
	};
	const auto start = input_at(0);
	const auto mask = input_at(1);
	const auto data = input_at(2);
	const Value active(&entry.AppendNewInst(Op::INotEqual32, {mask, Value(0u)}));
	auto& cursor = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	auto& payload = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
	cursor.AddPhiOperand(&entry, start);
	payload.AddPhiOperand(&entry, Value(0u));
	const bool dormant_test = variant == Variant::Dormant || variant == Variant::DormantRead ||
	                          variant == Variant::DormantVote || variant == Variant::DormantEscapes || variant == Variant::Revived;
	auto live_mask = active;
	Inst* mask_phi = nullptr;
	if (dormant_test) {
		mask_phi = &header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U1));
		mask_phi->AddPhiOperand(&entry, active);
		live_mask = Value(mask_phi);
		payload.SetArg(0, data);
	}
	const Value unfinished(&test.AppendNewInst(Op::INotEqual32, {Value(&payload), Value(77u)}));
	const Value live(&test.AppendNewInst(Op::LogicalAnd, {live_mask, unfinished}));
	auto contributed = live;
	if (variant == Variant::DormantVote || variant == Variant::DormantEscapes) {
		const Value coordinate(&test.AppendNewInst(Op::BitCastF32U32, {data}));
		const Value ordered(&test.AppendNewInst(Op::FPOrdEqual32, {coordinate, coordinate}));
		if (variant == Variant::DormantVote) {
			contributed = Value(&test.AppendNewInst(Op::LogicalOr, {live, ordered}));
		} else {
			const Value observable(&test.AppendNewInst(Op::SelectU32, {ordered, data, Value(1u)}));
			exit.AppendNewInst(Op::ReferenceU32, {observable});
		}
	}
	test.condition = variant == Variant::NoBallot ? Value(true) : Value(&test.AppendNewInst(Op::ConditionRef, {contributed},
	                       static_cast<uint64_t>(CFG::BranchCondition::ExecNonZero)));
	if (variant == Variant::OuterLane || variant == Variant::ScalarLane) {
		const Value lane(&entry.AppendNewInst(Op::GetBuiltin,
		                                     {Value(static_cast<uint32_t>(StageInputKind::LocalInvocationIndex)), Value(0u)}));
		const Value low(&entry.AppendNewInst(Op::ULessThan32, {lane, Value(32u)}));
		auto& join = *program.block_storage.emplace_back(std::make_unique<Block>());
		join.id = program.blocks.size();
		program.blocks.push_back(&join);
		link(exit, join);
		entry.terminator.kind = CFG::TerminatorKind::ConditionalBranch;
		entry.terminator.false_block = &join;
		entry.terminator.merge_block = &join;
		entry.condition = variant == Variant::OuterLane ? low :
		    Value(&entry.AppendNewInst(Op::ConditionRef, {low}, static_cast<uint64_t>(CFG::BranchCondition::ScalarInstruction)));
		entry.AddBranch(&join);
	}
	auto& memory = program.memory_info.emplace_back();
	memory.kind = ResourceKind::IndirectBuffer;
	memory.idxen = true;
	memory.coherent = variant == Variant::Coherent;
	const Value resource(&latch.AppendNewInst(Op::GetBufferResource,
	                                         {Value(0u), Value(0u), Value(1u), Value(1u << 12u)}));
	const auto read_mask = variant == Variant::OtherMask ? active : live;
	const Value loaded(&latch.AppendNewInst(Op::LoadBufferU32,
	                                       {resource, Value(&cursor), Value(0u), Value(0u), read_mask}));
	auto next_payload = loaded;
	if (dormant_test) {
		const Value coordinate(&latch.AppendNewInst(Op::BitCastF32U32, {data}));
		const Value ordered(&latch.AppendNewInst(Op::FPOrdEqual32, {coordinate, coordinate}));
		next_payload = Value(&latch.AppendNewInst(Op::SelectU32, {ordered, data, loaded}));
		if (variant == Variant::DormantRead)
			latch.AppendNewInst(Op::LoadBufferU32, {resource, Value(0u), Value(0u), Value(0u), Value(true)});
		mask_phi->AddPhiOperand(&latch, variant == Variant::Revived ? Value(true) : live);
	}
	payload.AddPhiOperand(&latch, Value(&latch.AppendNewInst(Op::SelectU32, {live, next_payload, Value(&payload)})));
	const Value increment(&latch.AppendNewInst(Op::IAdd32, {Value(&cursor), Value(1u)}));
	const Value step(&latch.AppendNewInst(Op::SelectU32, {live, increment, Value(&cursor)}));
	cursor.AddPhiOperand(&latch, step);
	exit.AppendNewInst(Op::ReferenceU32, {Value(&cursor)});
	if (variant == Variant::EscapeStep) {
		auto& carried = header.AppendNewInst(Op::Phi, {}, static_cast<uint64_t>(Type::U32));
		carried.AddPhiOperand(&entry, Value(0u));
		carried.AddPhiOperand(&latch, step);
		exit.AppendNewInst(Op::ReferenceU32, {Value(&carried)});
	}
	if (variant == Variant::Barrier) latch.AppendNewInst(Op::Barrier);
	if (variant == Variant::Atomic) {
		latch.AppendNewInst(Op::BufferAtomicIAdd32, {resource, Value(&cursor), Value(0u), Value(1u), Value(0u), live});
	}

	const auto guards = GuardComputeLoops(program, limit, variant == Variant::TwoWaves, {});
	Result result {};
	result.rewritten = cursor.Arg(1) != step;
	result.guarded = guards.guarded == 1;
	if (!result.guarded || variant == Variant::OuterLane || variant == Variant::ScalarLane ||
	    variant == Variant::OtherMask || variant == Variant::EscapeStep || variant == Variant::NoBallot) return result;

	std::map<const Inst*, Wave> values;
	values[start.Instruction()] = input.start;
	values[mask.Instruction()] = input.active;
	values[data.Instruction()] = input.data;
	values[active.Instruction()] = input.active;
	const auto read = [&](Value value, size_t lane) -> uint32_t {
		if (const auto* inst = value.TryInstruction()) return values.at(inst)[lane];
		return value.GetType() == Type::U1 ? value.U1() : value.U32();
	};
	const auto run = [&](const Block& block, const Block* from) {
		std::map<const Inst*, Wave> phis;
		for (auto& inst: block) {
			if (inst.GetOpcode() != Op::Phi) break;
			const auto val = inst.Arg(inst.PhiBlock(0) == from ? 0 : 1);
			for (size_t i = 0; i < result.cursor.size(); ++i) phis[&inst][i] = read(val, i);
		}
		for (const auto& [inst, wave]: phis) values[inst] = wave;
		for (auto& inst: block) {
			const auto op = inst.GetOpcode();
			if (op == Op::Phi || op == Op::GetBufferResource || op == Op::Barrier || op == Op::BufferAtomicIAdd32) continue;
			auto& output = values[&inst];
			if (op == Op::ConditionRef) {
				const auto& predicate = values.at(inst.Arg(0).Instruction());
				switch (inst.Flags<CFG::BranchCondition>()) {
					case CFG::BranchCondition::ExecNonZero:
						output.fill(std::ranges::any_of(predicate, [](uint32_t v) { return v != 0; }));
						break;
					case CFG::BranchCondition::ScalarInstruction: output = predicate; break;
					default: throw std::logic_error("unsupported vote in fixed loop test");
				}
				continue;
			}
			for (size_t i = 0; i < output.size(); ++i) {
				const auto arg = [&](size_t n) { return read(inst.Arg(n), i); };
				switch (op) {
					case Op::FPOrdEqual32: {
						const auto left = std::bit_cast<float>(arg(0));
						const auto right = std::bit_cast<float>(arg(1));
						output[i] = !std::isnan(left) && !std::isnan(right) && left == right;
						break;
					}
					case Op::BitCastU32F32:
					case Op::BitCastF32U32: output[i] = arg(0); break;
					case Op::BitwiseAnd32: output[i] = arg(0) & arg(1); break;
					case Op::LoadBufferU32: output[i] = arg(4) && arg(1) == 0 ? 77u : 0u; break;
					case Op::SelectU32: output[i] = arg(0) ? arg(1) : arg(2); break;
					case Op::IAdd32: output[i] = arg(0) + arg(1); break;
					case Op::ISub32: output[i] = arg(0) - arg(1); break;
					case Op::IEqual32: output[i] = arg(0) == arg(1); break;
					case Op::INotEqual32: output[i] = arg(0) != arg(1); break;
					case Op::ULessThan32: output[i] = arg(0) < arg(1); break;
					case Op::ULessThanEqual32: output[i] = arg(0) <= arg(1); break;
					case Op::UGreaterThanEqual32: output[i] = arg(0) >= arg(1); break;
					case Op::LogicalNot: output[i] = !arg(0); break;
					case Op::LogicalAnd: output[i] = arg(0) && arg(1); break;
					case Op::LogicalOr: output[i] = arg(0) || arg(1); break;
					case Op::LogicalXor: output[i] = static_cast<bool>(arg(0)) != static_cast<bool>(arg(1)); break;
					case Op::LoopGuardMark: result.marks += arg(0); break;
					default: throw std::logic_error("unsupported opcode in fixed loop test");
				}
			}
		}
	};
	const Block* pred = &entry;
	while (result.turns <= limit) {
		run(header, pred);
		run(test, &header);
		if (read(test.condition, 0) == 0) break;
		run(latch, &test);
		pred = &latch;
		++result.turns;
	}
	result.cursor = values.at(&cursor);
	result.payload = values.at(&payload);
	return result;
}

}

bool TestFixedLoopTail() {
	Inputs input {};
	input.start.fill(23);
	input.active.fill(1);
	for (const auto variant: {Variant::Plain, Variant::Coherent, Variant::Atomic, Variant::Barrier, Variant::TwoWaves}) {
		const auto r = Run(input, variant);
		const bool enabled = variant == Variant::Plain;
		if (r.rewritten != enabled || r.turns != (enabled ? 2u : 16u) || r.marks != 64u ||
		    !std::ranges::all_of(r.cursor, [](uint32_t i) { return i == 39u; })) return false;
	}
	for (const auto variant: {Variant::OuterLane, Variant::ScalarLane, Variant::OtherMask, Variant::EscapeStep, Variant::NoBallot}) {
		const auto r = Run(input, variant);
		if (!r.guarded || r.rewritten) return false;
	}
	for (const auto payload: {0x7fc00031u, 0xffc0abcdu, 0x7f800001u, 0xff801234u}) {
		Inputs masked {};
		masked.start.fill(23);
		masked.active[0] = 1;
		masked.data[63] = payload;
		for (const auto variant: {Variant::Dormant, Variant::DormantRead, Variant::DormantVote, Variant::DormantEscapes}) {
			const auto r = Run(masked, variant);
			if (!r.guarded || !r.rewritten || r.turns != (variant == Variant::Dormant ? 2u : 16u) ||
			    r.cursor[0] != 39u || r.cursor[63] != 23u || r.payload[63] != payload || r.marks != 64u) return false;
		}
		masked.active[63] = 1;
		const auto active_nan = Run(masked, Variant::Dormant);
		if (active_nan.turns != 16u || active_nan.cursor[63] != 39u || active_nan.marks != 64u) return false;
		masked.active[63] = 0;
		const auto revived = Run(masked, Variant::Revived);
		if (revived.turns != 16u || revived.cursor[63] != 38u || revived.marks != 64u) return false;
	}
	input.start[63] = UINT32_MAX - 3u;
	const auto reference = Run(input, Variant::TwoWaves);
	const auto mixed = Run(input, Variant::Plain);
	if (mixed.turns >= reference.turns || mixed.cursor != reference.cursor || mixed.payload != reference.payload ||
	    mixed.marks != reference.marks || mixed.cursor[63] != 1u || mixed.payload[63] != 77u) return false;
	input.active.fill(0);
	input.active[63] = 1;
	const auto wrapped = Run(input, Variant::Plain);
	return wrapped.turns == 5u && wrapped.marks == 0u && wrapped.cursor[63] == 1u && wrapped.cursor[0] == 23u;
}
