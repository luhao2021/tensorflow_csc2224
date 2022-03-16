#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_

// #include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
// #include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

llvm::Value* EmitCusToF32(llvm::Value* cus_value, llvm::IRBuilder<>* b);

StatusOr<llvm::Value*> EmitF32ToCus(llvm::Value* f32_value,
                                    llvm::IRBuilder<>* b);
#define CUS_UNARY_OP_HEADER(op) \
  llvm::Value* EmitCus##op(llvm::Value* cus_value, llvm::IRBuilder<>* b);
#define CUS_BINARY_OP_HEADER(op)                               \
  llvm::Value* EmitCus##op(llvm::Value* lhs, llvm::Value* rhs, \
                           llvm::IRBuilder<>* b)

CUS_UNARY_OP_HEADER(Neg);
CUS_UNARY_OP_HEADER(Log);
CUS_UNARY_OP_HEADER(Exp);
CUS_UNARY_OP_HEADER(Expm1);
CUS_UNARY_OP_HEADER(Log1p);
CUS_UNARY_OP_HEADER(Cos);
CUS_UNARY_OP_HEADER(Sin);
CUS_UNARY_OP_HEADER(Tanh);
CUS_UNARY_OP_HEADER(Sqrt);
CUS_UNARY_OP_HEADER(Rsqrt);
CUS_UNARY_OP_HEADER(Cbrt);
CUS_UNARY_OP_HEADER(Floor);
CUS_UNARY_OP_HEADER(Ceil);
CUS_UNARY_OP_HEADER(Abs);
CUS_UNARY_OP_HEADER(RoundNearestAfz);
CUS_UNARY_OP_HEADER(Sign);
CUS_UNARY_OP_HEADER(IsFinite);

CUS_BINARY_OP_HEADER(Add);
CUS_BINARY_OP_HEADER(Sub);
CUS_BINARY_OP_HEADER(Mul);
CUS_BINARY_OP_HEADER(Div);
CUS_BINARY_OP_HEADER(Eq);
CUS_BINARY_OP_HEADER(Ne);
CUS_BINARY_OP_HEADER(Lt);
CUS_BINARY_OP_HEADER(Gt);
CUS_BINARY_OP_HEADER(Le);
CUS_BINARY_OP_HEADER(Ge);
CUS_BINARY_OP_HEADER(Max);
CUS_BINARY_OP_HEADER(Min);
CUS_BINARY_OP_HEADER(Pow);
CUS_BINARY_OP_HEADER(Atan2);


}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CUS_RELATED_FUNCTIONS_H_
