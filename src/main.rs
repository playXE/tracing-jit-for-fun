use cranelift::prelude::*;
use cranelift_module::{DataContext, Linkage, Module};
use cranelift_simplejit::{SimpleJITBackend, SimpleJITBuilder};
use std::collections::HashMap;
/// The basic JIT class.
pub struct JIT {
    /// The function builder context, which is reused across multiple
    /// FunctionBuilder instances.
    builder_context: FunctionBuilderContext,

    /// The main Cranelift context, which holds the state for codegen. Cranelift
    /// separates this from `Module` to allow for parallel compilation, with a
    /// context per thread, though this isn't in the simple demo here.
    ctx: codegen::Context,

    /// The data context, which is to data objects what `ctx` is to functions.
    data_ctx: DataContext,

    /// The module, with the simplejit backend, which manages the JIT'd
    /// functions.
    module: Module<SimpleJITBackend>,
}
impl JIT {
    /// Create a new `JIT` instance.
    pub fn new() -> Self {
        let builder = SimpleJITBuilder::new(cranelift_module::default_libcall_names());
        let module = Module::new(builder);
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            data_ctx: DataContext::new(),
            module,
        }
    }
}

macro_rules! log  {
    ($($t: tt)*) => {
        if LOG {
            println!($($t)*);
        }
    };
}

pub const OP_PUSH: i32 = 0;
pub const OP_ADD: i32 = 1;
pub const OP_JUMP: i32 = 2;
pub const OP_GT: i32 = 3;
pub const OP_HALT: i32 = 4;
pub const OP_POP: i32 = 5;
pub const TRACE_INSTR: i32 = 0;
pub const TRACE_GT_JUMP: i32 = 1;
pub const TRACE_GT_NJUMP: i32 = 2;
pub const TRACE_ENTER_TRACE: i32 = 3;

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum Op {
    Movi(u8, i32),
    Gt(u8, i32, usize),
    Jump(usize),
    Add(u8, i32),
    Ret(u8),
}

trait Interpreter {
    fn pc(&self) -> usize;
    fn pc_mut(&mut self) -> &mut usize;
    fn stack(&self) -> &[i32];
    fn stack_mut(&mut self) -> &mut Vec<i32>;
    fn code(&self) -> &[Op];

    fn run_movi(&mut self) {
        match self.code()[self.pc()] {
            Op::Movi(dst, i) => {
                self.stack_mut()[dst as usize] = i;
            }
            _ => unreachable!(),
        }
        *self.pc_mut() += 1;
    }

    fn run_gt(&mut self) {
        match self.code()[self.pc()] {
            Op::Gt(reg, x, target) => {
                if self.stack()[reg as usize] > x {
                    *self.pc_mut() = target;
                } else {
                    *self.pc_mut() += 1;
                }
            }
            _ => unreachable!(),
        }
    }

    fn run_add(&mut self) {
        match self.code()[self.pc()] {
            Op::Add(reg, i) => {
                self.stack_mut()[reg as usize] += i;
            }
            _ => unreachable!(),
        }
        *self.pc_mut() += 1;
    }
    fn run_jump(&mut self) {
        match self.code()[self.pc()] {
            Op::Jump(target) => {
                *self.pc_mut() = target;
            }
            _ => unreachable!(),
        }
    }

    fn interpret(&mut self) -> i32 {
        loop {
            //log!("{}: {:?} ", self.pc(), self.code()[self.pc()]);
            let ins = self.code()[self.pc()];
            match ins {
                Op::Add { .. } => self.run_add(),
                Op::Jump { .. } => self.run_jump(),
                Op::Ret(r) => return self.stack()[r as usize],
                Op::Gt { .. } => self.run_gt(),
                Op::Movi { .. } => self.run_movi(),
            }
        }
    }
}

struct LoopInfo {
    hotness: usize,
    fails: usize,
    trace_id: usize,
    blacklisted: bool,
    trace: Vec<(Trace, usize)>,
    executable_trace: Option<extern "C" fn(*mut i32, *mut usize) -> i32>,
}

pub struct TracingInterpreter<'a> {
    loops: HashMap<(usize, usize), LoopInfo>,
    code: Vec<Op>,
    stack: Vec<i32>,
    recording: bool,
    trace_id: usize,
    pc: usize,
    jit: &'a mut JIT,
}

impl<'a> TracingInterpreter<'a> {
    pub fn new(jit: &'a mut JIT, code: Vec<Op>) -> Self {
        Self {
            loops: HashMap::new(),
            code,
            stack: vec![0, 0, 0, 0, 0, 0],
            recording: false,
            trace_id: 0,
            pc: 0,
            jit,
        }
    }
    fn translate_trace(&mut self, info: &LoopInfo) -> extern "C" fn(*mut i32, *mut usize) -> i32 {
        let int = self.jit.module.target_config().pointer_type();
        self.jit.ctx.func.signature.params.push(AbiParam::new(int));
        self.jit.ctx.func.signature.params.push(AbiParam::new(int));
        self.jit
            .ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(types::I32));
        self.jit.ctx.set_disasm(true);
        let mut builder =
            FunctionBuilder::new(&mut self.jit.ctx.func, &mut self.jit.builder_context); // Create the entry block, to start emitting code in.
        let entry_block = builder.create_block();

        // Since this is the entry block, add block parameters corresponding to
        // the function's parameters.
        builder.append_block_params_for_function_params(entry_block);

        // Tell the builder to emit code in this block.
        builder.switch_to_block(entry_block);
        let stack = builder.block_params(entry_block)[0];
        let pc = builder.block_params(entry_block)[1];
        // And, tell the builder that this block will have no further
        // predecessors. Since it's the entry block, it won't have any
        // predecessors.
        builder.seal_block(entry_block);
        let loop_block = builder.create_block();
        builder.ins().fallthrough(loop_block, &[]);

        builder.switch_to_block(loop_block);
        let load_reg = |builder: &mut FunctionBuilder, r: u8| {
            builder
                .ins()
                .load(types::I32, MemFlags::new(), stack, r as i32 * 4)
        };

        let store_reg = |builder: &mut FunctionBuilder, r: u8, val: Value| {
            builder
                .ins()
                .store(MemFlags::new(), val, stack, r as i32 * 4);
        };
        let _update_pc = |builder: &mut FunctionBuilder, to: usize| {
            let prev = builder.ins().load(types::I64, MemFlags::new(), pc, 0);
            let new = builder.ins().iadd_imm(prev, to as i64);
            builder.ins().store(MemFlags::new(), new, pc, 0);
        };
        let mut pc_to_bb = HashMap::new();

        let mut gen_guards_slow_path: Vec<Box<dyn FnOnce(&mut FunctionBuilder)>> = vec![];
        for trace_step in info.trace.iter() {
            pc_to_bb.insert(trace_step.1, builder.current_block().unwrap());
            match &trace_step.0 {
                Trace::Instr(op) => match op {
                    Op::Jump(t) => {
                        /*if let Some(bb) = pc_to_bb.get(t) {
                            builder.ins().jump(*bb, &[]);
                            let next = builder.create_block();
                            builder.switch_to_block(next);
                        } else {
                            finalize_jumps.push((*t, builder.current_block().unwrap()));
                            let next = builder.create_block();
                            builder.switch_to_block(next);
                        }*/
                        //let x = builder.ins().iconst(types::I64, *t as i64);
                        //builder.ins().store(MemFlags::new(), x, pc, 0);
                    }
                    Op::Add(r, imm) => {
                        let x = load_reg(&mut builder, *r);
                        let y = builder.ins().iadd_imm(x, *imm as i64);
                        store_reg(&mut builder, *r, y);
                        //update_pc(&mut builder, 1);
                    }
                    Op::Movi(r, imm) => {
                        let x = builder.ins().iconst(types::I64, *imm as i64);
                        store_reg(&mut builder, *r, x);
                        // update_pc(&mut builder, 1);
                    }
                    _ => unreachable!(),
                },
                Trace::GuardGtJump(r, imm, pc_) => {
                    let x = load_reg(&mut builder, *r);
                    let imm = builder.ins().iconst(types::I32, *imm as i64);
                    let next = builder.create_block();
                    let fail = builder.create_block();
                    let pc_ = *pc_;
                    builder
                        .ins()
                        .br_icmp(IntCC::SignedLessThanOrEqual, x, imm, fail, &[]);
                    builder.ins().jump(next, &[]);
                    builder.switch_to_block(next);
                    gen_guards_slow_path.push(Box::new(move |builder: &mut FunctionBuilder| {
                        builder.switch_to_block(fail);
                        let imm = builder.ins().iconst(types::I64, pc_ as i64);
                        builder.ins().store(MemFlags::new(), imm, pc, 0);
                        let ret_code = builder.ins().iconst(types::I32, 1);
                        builder.ins().return_(&[ret_code]);
                    }));
                }
                Trace::GuardGtNJump(r, imm, pc_) => {
                    let x = load_reg(&mut builder, *r);
                    let imm = builder.ins().iconst(types::I32, *imm as i64);
                    let next = builder.create_block();
                    let fail = builder.create_block();
                    builder
                        .ins()
                        .br_icmp(IntCC::SignedGreaterThan, x, imm, fail, &[]);
                    let pc_ = *pc_;
                    builder.ins().jump(next, &[]);
                    builder.switch_to_block(next);
                    gen_guards_slow_path.push(Box::new(move |builder: &mut FunctionBuilder| {
                        builder.switch_to_block(fail);
                        let imm = builder.ins().iconst(types::I64, pc_ as i64);
                        builder.ins().store(MemFlags::new(), imm, pc, 0);
                        let ret_code = builder.ins().iconst(types::I32, 1);
                        builder.ins().return_(&[ret_code]);
                    }));
                }
                Trace::EnterTrace { .. } => {}
            }
        }
        //let end = builder.create_block();
        //builder.ins().fallthrough(end, &[]);
        //builder.switch_to_block(end);
        builder.ins().jump(loop_block, &[]);
        for slow in gen_guards_slow_path {
            slow(&mut builder);
        }

        builder.seal_all_blocks();
        builder.finalize();
        let disp = builder.func.display(None).to_string();
        let id = self
            .jit
            .module
            .declare_function(
                &format!("trace_{}", info.trace_id),
                Linkage::Export,
                &self.jit.ctx.func.signature,
            )
            .unwrap();

        let r = self
            .jit
            .module
            .define_function(
                id,
                &mut self.jit.ctx,
                &mut codegen::binemit::NullTrapSink {},
            )
            .map_err(|e| e.to_string());
        if let Err(r) = r {
            panic!(r);
        }
        self.jit.module.clear_context(&mut self.jit.ctx);
        self.jit.module.finalize_definitions();
        // We can now retrieve a pointer to the machine code.
        let code = self.jit.module.get_finalized_function(id);
        log!(
            "Generated Cranelift IR from trace at {:p}: \n{}",
            code,
            disp
        );
        unsafe { std::mem::transmute(code) }
    }
}

impl Interpreter for TracingInterpreter<'_> {
    fn stack(&self) -> &[i32] {
        &self.stack
    }

    fn stack_mut(&mut self) -> &mut Vec<i32> {
        &mut self.stack
    }

    fn pc(&self) -> usize {
        self.pc
    }

    fn pc_mut(&mut self) -> &mut usize {
        &mut self.pc
    }

    fn code(&self) -> &[Op] {
        &self.code
    }

    fn run_jump(&mut self) {
        let old_pc = self.pc;
        let new_pc = match self.code()[self.pc] {
            Op::Jump(pc) => pc,
            _ => unreachable!(),
        };
        let this = unsafe { &mut *(self as *const Self as *mut Self) };
        if new_pc < old_pc {
            if self.loops.contains_key(&(new_pc, old_pc)) {
                let info = self.loops.get_mut(&(new_pc, old_pc)).unwrap();
                if info.blacklisted {
                    self.pc = new_pc;
                    return;
                }
                info.hotness += 1;
                if let Some(trace) = info.executable_trace {
                    self.pc = new_pc;
                    log!("stack ptr {:p}, pc ptr {:p}", self.stack.as_ptr(), &self.pc);
                    let is_ok = trace(self.stack.as_mut_ptr(), &mut self.pc);
                    if is_ok != 0 {
                        log!("Guard failed, leaving trace for interpreter execution ({} traces left until blacklisting)",10 - info.fails);
                        info.fails += 1;
                        if info.fails == 10 {
                            log!("Blacklisting trace");
                            info.blacklisted = true;
                        }
                        return;
                    }
                } else if info.hotness > 1000 && info.executable_trace.is_none() {
                    if !self.recording {
                        log!("Found new loop from {}->{}", new_pc, old_pc);
                        self.recording = true;
                        self.pc = new_pc;
                        let mut trace = vec![];
                        let mut recording = RecordingInterpreter {
                            stack: &mut self.stack,
                            pc: &mut self.pc,
                            done: false,
                            trace: &mut trace,
                            code: &self.code,
                            end_of_trace: old_pc,
                            trace_is_too_big: false,
                        };

                        log!("Trace recording interpreter started at pc = {} until (included) pc = {}",new_pc,old_pc);
                        recording.interpret();
                        if recording.done {
                            self.recording = false;
                            info.trace = trace;
                            info.trace_id = self.trace_id;
                            let f = this.translate_trace(info);
                            self.trace_id += 1;
                            info.executable_trace = Some(f);
                            log!("Now jumping to compiled trace!");
                            self.run_jump();
                            return;
                        } else if recording.trace_is_too_big {
                            info.fails += 1;
                            log!("Trace is too big, {} traces left until blacklisting ({} traces failed)",10 - info.fails,info.fails);
                            if info.fails == 10 {
                                log!("Generated too big trace 10 times,blacklisting");
                                info.blacklisted = true;
                            }
                        }
                    }
                }
            } else {
                self.loops.insert(
                    (new_pc, old_pc),
                    LoopInfo {
                        executable_trace: None,
                        trace: vec![],
                        trace_id: 0,
                        hotness: 1,
                        blacklisted: false,
                        fails: 0,
                    },
                );
                self.recording = false;
            }
        }
        self.pc = new_pc;
    }
}
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub enum Trace {
    Instr(Op),
    GuardGtJump(u8, i32, usize /* pc */),
    GuardGtNJump(u8, i32, usize /* pc */),
    EnterTrace,
}

pub struct RecordingInterpreter<'a> {
    pc: &'a mut usize,
    stack: &'a mut Vec<i32>,
    code: &'a [Op],
    trace: &'a mut Vec<(Trace, usize)>,
    end_of_trace: usize,
    trace_is_too_big: bool,
    done: bool,
}

impl Interpreter for RecordingInterpreter<'_> {
    fn stack(&self) -> &[i32] {
        &self.stack
    }

    fn stack_mut(&mut self) -> &mut Vec<i32> {
        &mut self.stack
    }

    fn pc(&self) -> usize {
        *self.pc
    }

    fn pc_mut(&mut self) -> &mut usize {
        &mut self.pc
    }

    fn code(&self) -> &[Op] {
        &self.code
    }

    fn run_movi(&mut self) {
        self.trace
            .push((Trace::Instr(self.code()[self.pc()]), self.pc()));
        match self.code()[self.pc()] {
            Op::Movi(r, x) => self.stack_mut()[r as usize] = x,
            _ => unreachable!(),
        }
        *self.pc_mut() += 1;
    }

    fn run_gt(&mut self) {
        match self.code()[self.pc()] {
            Op::Gt(r, x, target) => {
                if self.stack()[r as usize] > x {
                    self.trace
                        .push((Trace::GuardGtJump(r, x, self.pc()), self.pc()));
                    self.trace.push((Trace::Instr(Op::Jump(target)), self.pc()));
                    *self.pc_mut() = target;
                } else {
                    self.trace
                        .push((Trace::GuardGtNJump(r, x, self.pc()), self.pc()));
                    self.trace
                        .push((Trace::Instr(Op::Jump(self.pc() + 1)), self.pc()));
                    *self.pc_mut() += 1;
                }
            }
            _ => unreachable!(),
        }
    }

    fn run_add(&mut self) {
        self.trace
            .push((Trace::Instr(self.code()[self.pc()]), self.pc()));
        match self.code()[self.pc()] {
            Op::Add(r, i) => {
                self.stack_mut()[r as usize] += i;
            }
            _ => unreachable!(),
        }
        *self.pc_mut() += 1;
    }
    fn run_jump(&mut self) {
        let end_of_trace = self.pc() == self.end_of_trace;
        self.trace
            .push((Trace::Instr(self.code()[self.pc()]), self.pc()));
        if end_of_trace {
            self.done = true;
            return;
        }
        match self.code()[self.pc()] {
            Op::Jump(target) => {
                *self.pc_mut() = target;
            }
            _ => unreachable!(),
        }
    }
    fn interpret(&mut self) -> i32 {
        loop {
            let ins = self.code()[self.pc()];
            log!("Trace #{} {}: {:?}", self.trace.len(), self.pc(), ins);
            match ins {
                Op::Add { .. } => self.run_add(),
                Op::Jump { .. } => self.run_jump(),
                Op::Ret(r) => return self.stack()[r as usize],
                Op::Gt { .. } => self.run_gt(),
                Op::Movi { .. } => self.run_movi(),
            }
            if self.trace.len() >= 50 {
                self.trace_is_too_big = true;
                self.done = false;
                return 0;
            }
            if self.done {
                log!("Trace finished");
                return 0;
            }
        }
    }
}

struct SimpleInterpreter {
    stack: Vec<i32>,
    pc: usize,
    code: Vec<Op>,
}
impl Interpreter for SimpleInterpreter {
    fn stack(&self) -> &[i32] {
        &self.stack
    }

    fn stack_mut(&mut self) -> &mut Vec<i32> {
        &mut self.stack
    }

    fn pc(&self) -> usize {
        self.pc
    }

    fn pc_mut(&mut self) -> &mut usize {
        &mut self.pc
    }

    fn code(&self) -> &[Op] {
        &self.code
    }
}

impl SimpleInterpreter {
    pub fn new(code: Vec<Op>) -> Self {
        Self {
            code,
            stack: vec![0, 0, 0, 0, 0, 0],
            pc: 0,
        }
    }
}
const LOG: bool = false;

fn main() {
    let code = vec![
        Op::Movi(0, 0),
        Op::Gt(0, 500000, 4),
        Op::Add(0, 1),
        Op::Jump(1),
        Op::Ret(0),
    ];
    let two_loops = vec![
        Op::Movi(1, 0),
        Op::Gt(1, 500000, 4),
        Op::Add(1, 1),
        Op::Jump(1),
        Op::Gt(1, 10000000, 7),
        Op::Add(1, 2),
        Op::Jump(4),
        Op::Ret(1),
    ];
    let code = two_loops;
    let nested_loops = vec![
        Op::Movi(0, 0),
        Op::Gt(0, 3000000, 8),
        Op::Movi(1, 0),
        Op::Gt(1, 15000, 6),
        Op::Add(1, 1),
        Op::Jump(3),
        Op::Add(0, 2),
        Op::Jump(1),
        Op::Ret(0),
    ];
    //let code = nested_loops;
    let mut interp = SimpleInterpreter::new(code.clone());
    let interp_start = std::time::Instant::now();
    interp.interpret();
    let interp_end = interp_start.elapsed();
    let mut jit = JIT::new();
    let mut tracing = TracingInterpreter::new(&mut jit, code.clone());
    let tracing_interp_start = std::time::Instant::now();
    tracing.interpret();
    let tracing_interp_end = tracing_interp_start.elapsed();

    println!(
        "Interpreter with tracing JIT executed in {}ms ({}ns)",
        tracing_interp_end.as_millis(),
        tracing_interp_end.as_nanos()
    );
    println!(
        "Interpreter without tracing JIT executed in {}ms ({}ns)",
        interp_end.as_millis(),
        interp_end.as_nanos()
    );
}
