#include "tensorflow/compiler/xla/service/cus_related_functions.h"

#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"

// #include "llvm/IRReader/IRReader.h"
// #include "llvm/Support/SourceMgr.h"
// #include "llvm/Linker/Linker.h"
// 
namespace xla {

llvm::Value* EmitCusToF32(llvm::Value* cus_value, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CastValueToF32", b->getFloatTy(), cus)
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {cus_value});
}

StatusOr<llvm::Value*> EmitF32ToCus(llvm::Value* f32_value,
                                    llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CastF32ToValue", cus, b->getFloatTy())
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {f32_value});
}

llvm::Value* EmitCusNeg(llvm::Value* cus_value, llvm::IRBuilder<>* b) {
  llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();
  llvm::Type* cus = llvm_ir::getCusTy(module->getContext());
  llvm::Value* func =
      module->getOrInsertFunction("CusNeg", cus, cus)
          .getCallee();
  return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {cus_value});
}

#define CUS_BINARY_OP(op)                                                   \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs,              \
                           llvm::IRBuilder<>* b) {                          \
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();   \
    llvm::Type* cus = llvm_ir::getCusTy(module->getContext());        \
    llvm::Value* func =                                                     \
        module->getOrInsertFunction("Cus" #op, cus, cus, cus).getCallee();  \
    return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {lhs, rhs}); \
  }

CUS_BINARY_OP(Add);
CUS_BINARY_OP(Sub);
CUS_BINARY_OP(Mul);
CUS_BINARY_OP(Div);

#define CUS_COMPARE(op)                                              \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs,              \
                           llvm::IRBuilder<>* b) {                          \
    llvm::Module* module = b->GetInsertBlock()->getParent()->getParent();   \
    llvm::Type* cus = llvm_ir::getCusTy(module->getContext());        \
    llvm::Value* func =                                                     \
        module                                                              \
            ->getOrInsertFunction(                                          \
                "Cus" #op, b->getIntNTy(8), cus, cus) \
            .getCallee();                                                   \
    return b->CreateCall(llvm::dyn_cast<llvm::Function>(func), {lhs, rhs}); \
  }

CUS_COMPARE(Eq);
CUS_COMPARE(Ne);
CUS_COMPARE(Lt);
CUS_COMPARE(Gt);
CUS_COMPARE(Le);
CUS_COMPARE(Ge);

}  // namespace xla