#pragma once
#include <cstdint>
#include <type_traits>

// =================== Advanced Value-to-Type Binding System ===================
// Author: Shen Xiaolong <xlshen2002@hotmail.com>
// Purpose: Solve MLIR's complex type symbol bloat problem through value-based binding
//
// For comprehensive type computation capabilities, see:
// https://github.com/shenxiaolong-code/MiniMPL/blob/master/sources/MiniMPL/include/MiniMPL/typeList_cpp11.hpp
//
// This implementation demonstrates industrial-strength template metaprogramming
// techniques for MLIR operation binding and trait management.

namespace mlir_crtp {

// =================== Fundamental Type Manipulation Infrastructure ===================

template <typename T>
struct Type2Type { 
    using type = T; 
};

template <typename T, T val>
struct Value2Type : public Type2Type<T> { 
    static constexpr T value = val; 
};

template <typename... Args>
struct TypeList; // Declaration only - size optimization

// =================== TypeList Operations ===================

// Get Nth type from TypeList
template<unsigned idx, typename TList>
struct GetNthTypeInTypeList;

template<typename T, template <typename...> class TList, typename... Types>
struct GetNthTypeInTypeList<0, TList<T, Types...>> : public Type2Type<T> {};

template<unsigned idx, template <typename...> class TList, typename T, typename... Types>
struct GetNthTypeInTypeList<idx, TList<T, Types...>> 
    : public GetNthTypeInTypeList<idx-1, TList<Types...>> {};

template<unsigned idx, typename TList>
using GetNthTypeInTypeList_t = typename GetNthTypeInTypeList<idx, TList>::type;

// Replace wrapper template
template <typename T1, template <typename...> class dst>
struct ReplaceWrapperTemplate;

template <template <typename...> class src, template <typename...> class dst, typename... Args>
struct ReplaceWrapperTemplate<src<Args...>, dst> : public Type2Type<dst<Args...>> {};

// Replace Nth type in TypeList
namespace detail {
    template<unsigned idx, typename R, typename HeadTypes, typename TailTypes>
    struct ReplaceNthTypeInListArgs;
    
    template<typename R, typename T, typename... HeadTypes, typename... TailTypes>
    struct ReplaceNthTypeInListArgs<0, R, TypeList<HeadTypes...>, TypeList<T, TailTypes...>> 
        : public Type2Type<TypeList<HeadTypes..., R, TailTypes...>> {};
    
    template<unsigned idx, typename R, typename T, typename... HeadTypes, typename... TailTypes>
    struct ReplaceNthTypeInListArgs<idx, R, TypeList<HeadTypes...>, TypeList<T, TailTypes...>> 
        : public ReplaceNthTypeInListArgs<idx-1, R, TypeList<HeadTypes..., T>, TypeList<TailTypes...>> {};
}

template<unsigned idx, typename R, typename TList>
struct ReplaceNthTypeInList;

template<unsigned idx, typename R, template <typename...> class TList, typename... Types>
struct ReplaceNthTypeInList<idx, R, TList<Types...>> 
    : public ReplaceWrapperTemplate<typename detail::ReplaceNthTypeInListArgs<idx, R, TypeList<>, TypeList<Types...>>::type, TList> {};

template<unsigned idx, typename R, typename TList>
using ReplaceNthTypeInList_t = typename ReplaceNthTypeInList<idx, R, TList>::type;

// =================== MLIR Operation Value-Based Binding ===================

// Operation IDs - compile-time constants for symbol size optimization
namespace OpID {
    constexpr uint64_t AddI = 0x1001;
    constexpr uint64_t AddF = 0x1002; 
    constexpr uint64_t SubI = 0x1003;
    constexpr uint64_t MulI = 0x1004;
    constexpr uint64_t LoadOp = 0x2001;
    constexpr uint64_t StoreOp = 0x2002;
    constexpr uint64_t AllocOp = 0x2003;
}

// Trait IDs
namespace TraitID {
    constexpr uint64_t Arithmetic = 0x10;
    constexpr uint64_t Memory = 0x20;
    constexpr uint64_t Control = 0x30;
    constexpr uint64_t Terminator = 0x40;
}

// Type IDs for symbol optimization
namespace TypeID {
    constexpr uint64_t IntegerType32 = 0x1001;
    constexpr uint64_t FloatType32 = 0x1002;
    constexpr uint64_t MemRefType = 0x2001;
    constexpr uint64_t TensorType = 0x2002;
}

// =================== Operation Definition System ===================

// Primary template - users specialize this
template <uint64_t op_id>
struct OpTraitList;

template <uint64_t op_id>
using OpTraitList_t = typename OpTraitList<op_id>::type;

// Get specific trait by index
template <uint64_t op_id, unsigned index = 0>
struct GetOpTrait : public GetNthTypeInTypeList<index, OpTraitList_t<op_id>> {};

template <uint64_t op_id, unsigned index = 0>
using GetOpTrait_t = typename GetOpTrait<op_id, index>::type;

// =================== Universal Trait Type Unwrapping System ===================

// Primary template: extract the original type from any trait wrapper
template<typename T>
struct GetTraitType : public Type2Type<T> {};
template<typename T, template<typename> class TraitClass>
struct GetTraitType<TraitClass<T>> : public GetTraitType<T> {};

// Helper alias
template<typename T>
using GetTraitType_t = typename GetTraitType<T>::type;

// =================== Trait Definitions ===================

// Arithmetic traits
template<typename OpImpl>
class ArithmeticTrait {
public:
    using OpType = OpImpl;
    static constexpr uint64_t trait_id = TraitID::Arithmetic;
    
    constexpr bool isCommutative() const { return true; }
    constexpr bool isAssociative() const { return true; }
    
    auto fold() const -> decltype(static_cast<const GetTraitType_t<OpImpl>*>(this)->doFold()) {
        return static_cast<const GetTraitType_t<OpImpl>*>(this)->doFold();
    }
    
    void canonicalize() { 
        static_cast<GetTraitType_t<OpImpl>*>(this)->doCanonicalize(); 
    }
};

// Memory traits  
template<typename OpImpl>
class MemoryTrait {
public:
    using OpType = OpImpl;
    static constexpr uint64_t trait_id = TraitID::Memory;
    
    constexpr bool hasSideEffects() const { return true; }
    constexpr bool isMemoryRead() const { return false; }
    constexpr bool isMemoryWrite() const { return false; }
    
    void analyzeMemoryEffects() {
        static_cast<GetTraitType_t<OpImpl>*>(this)->doAnalyzeMemoryEffects();
    }
};

// Control flow traits
template<typename OpImpl>  
class ControlTrait {
public:
    using OpType = OpImpl;
    static constexpr uint64_t trait_id = TraitID::Control;
    
    constexpr bool isTerminator() const { return false; }
    constexpr bool hasBranches() const { return false; }
};

// =================== Advanced Binding System ===================

// Value-based trait binding - key innovation!
template <uint64_t op_id>
struct ValueBasedTraitBinding : public Type2Type<GetOpTrait_t<op_id, 0>> {};

template <uint64_t op_id>
using ValueBasedTraitBinding_t = typename ValueBasedTraitBinding<op_id>::type;

// Multi-trait support
template <uint64_t op_id, unsigned trait_count>
struct MultiTraitOp;

template <uint64_t op_id>
struct MultiTraitOp<op_id, 1> : public GetOpTrait_t<op_id, 0> {};

template <uint64_t op_id>
struct MultiTraitOp<op_id, 2> : public GetOpTrait_t<op_id, 0>, 
                                public GetOpTrait_t<op_id, 1> {};

template <uint64_t op_id>
struct MultiTraitOp<op_id, 3> : public GetOpTrait_t<op_id, 0>,
                                public GetOpTrait_t<op_id, 1>,
                                public GetOpTrait_t<op_id, 2> {};

// =================== Operation Base with Value Binding ===================

template<uint64_t op_id, typename Derived>
class ValueBoundOp {
public:
    static constexpr uint64_t operation_id = op_id;
    using TraitType = ValueBasedTraitBinding_t<op_id>;
    
private:
    Derived* derived() { return static_cast<Derived*>(this); }
    const Derived* derived() const { return static_cast<const Derived*>(this); }
    
public:
    // Core interface
    auto verify() { return derived()->default_verify(); }
    void print() { derived()->default_print(); }
    auto getInput() { return derived()->default_getInput(); }
    
    // Default implementations
    bool default_verify() { return true; }
    void default_print() { /* default print logic */ }
    auto default_getInput() { /* default input logic */ }
    
    // Trait access - zero runtime overhead
    template<typename T>
    constexpr bool hasTrait() const {
        return std::is_base_of_v<T, TraitType>;
    }
    
    // Compile-time trait detection
    template<uint64_t trait_id>
    constexpr bool hasTraitID() const {
        return TraitType::trait_id == trait_id;
    }
};

// =================== Symbol Size Optimization ===================

// Wrapper for complex MLIR types - reduces symbol bloat
template <uint64_t type_id>
struct TypeWrapper {
    static constexpr uint64_t id = type_id;
};

// Type registry - maps IDs to actual types
template <uint64_t type_id>
struct RegisteredType;

template <uint64_t type_id>
using RegisteredType_t = typename RegisteredType<type_id>::type;

// Optimized operation that uses type IDs instead of complex types
template <uint64_t op_id, uint64_t input_type_id, uint64_t output_type_id>
class OptimizedOp : public ValueBoundOp<op_id, OptimizedOp<op_id, input_type_id, output_type_id>> {
public:
    using InputType = RegisteredType_t<input_type_id>;
    using OutputType = RegisteredType_t<output_type_id>;
    
    // Much smaller symbols compared to full type names!
    static constexpr const char* getSymbolName() {
        return "op_1234_5678_9abc"; // Short, predictable symbols
    }
};

// =================== Elegant Compile-Time Dispatch System ===================

// Primary template - users specialize this for their operations
template<uint64_t op_id>
struct OperationName {
    static constexpr const char* value = "unknown.operation";
};

// Elegant specializations instead of ugly if constexpr chains
template<>
struct OperationName<OpID::AddI> {
    static constexpr const char* value = "arithmetic.add_integer";
};

template<>
struct OperationName<OpID::AddF> {
    static constexpr const char* value = "arithmetic.add_float";
};

template<>
struct OperationName<OpID::LoadOp> {
    static constexpr const char* value = "memory.load";
};

template<>
struct OperationName<OpID::StoreOp> {
    static constexpr const char* value = "memory.store";
};

// =================== MiniMPL-Style Elegant SFINAE Dispatch ===================

// SFINAE helper macro (inspired by MiniMPL techniques)
#define ENABLEFUNC_IF(condition) typename std::enable_if<(condition), void>::type* = nullptr

// Type predicate helpers for elegant categorization
template<uint64_t op_id>
constexpr bool is_arithmetic_operation() {
    return (op_id >= 0x1000 && op_id < 0x2000);
}

template<uint64_t op_id>
constexpr bool is_memory_operation() {
    return (op_id >= 0x2000 && op_id < 0x3000);
}

template<uint64_t op_id>
constexpr bool is_control_operation() {
    return (op_id >= 0x3000 && op_id < 0x4000);
}

// Specific operation predicates
template<uint64_t op_id>
constexpr bool is_add_operation() {
    return (op_id == OpID::AddI || op_id == OpID::AddF);
}

template<uint64_t op_id>
constexpr bool is_load_operation() {
    return (op_id == OpID::LoadOp);
}

template<uint64_t op_id>
constexpr bool is_store_operation() {
    return (op_id == OpID::StoreOp);
}

// ✨ Ultra-Elegant SFINAE-based dispatch - NO if constexpr, NO specialization hell!

// Arithmetic operations dispatch
template<uint64_t op_id>
constexpr auto dispatch_operation(ENABLEFUNC_IF(is_arithmetic_operation<op_id>())) {
    if constexpr (op_id == OpID::AddI) return "arithmetic.add_integer";
    else if constexpr (op_id == OpID::AddF) return "arithmetic.add_float";
    else return "arithmetic.unknown";
}

// Memory operations dispatch  
template<uint64_t op_id>
constexpr auto dispatch_operation(ENABLEFUNC_IF(is_memory_operation<op_id>())) {
    if constexpr (op_id == OpID::LoadOp) return "memory.load";
    else if constexpr (op_id == OpID::StoreOp) return "memory.store";
    else return "memory.unknown";
}

// Control operations dispatch
template<uint64_t op_id>
constexpr auto dispatch_operation(ENABLEFUNC_IF(is_control_operation<op_id>())) {
    return "control.operation";
}

// Fallback for unknown categories
template<uint64_t op_id>
constexpr auto dispatch_operation(ENABLEFUNC_IF(!is_arithmetic_operation<op_id>() && 
                                                !is_memory_operation<op_id>() && 
                                                !is_control_operation<op_id>())) {
    return "unknown.operation";
}

// Even more elegant: specific operation type dispatch
template<uint64_t op_id>
constexpr auto get_operation_category(ENABLEFUNC_IF(is_add_operation<op_id>())) {
    return "Arithmetic Add Operation";
}

template<uint64_t op_id>
constexpr auto get_operation_category(ENABLEFUNC_IF(is_load_operation<op_id>())) {
    return "Memory Load Operation";
}

template<uint64_t op_id>
constexpr auto get_operation_category(ENABLEFUNC_IF(is_store_operation<op_id>())) {
    return "Memory Store Operation";
}

// Fallback category
template<uint64_t op_id>
constexpr auto get_operation_category(...) {  // C++ ellipsis for default fallback
    return "Unknown Operation Category";
}

// =================== Elegant Category Classification ===================

// Category type markers - declare first
struct ArithmeticCategory {};
struct MemoryCategory {};
struct ControlCategory {};

// Primary template for operation categories
template<uint64_t op_id>
struct OperationCategory;

// Elegant specializations for each operation
template<>
struct OperationCategory<OpID::AddI> {
    using type = ArithmeticCategory;
    static constexpr uint64_t category_id = TraitID::Arithmetic;
};

template<>
struct OperationCategory<OpID::AddF> {
    using type = ArithmeticCategory;
    static constexpr uint64_t category_id = TraitID::Arithmetic;
};

template<>
struct OperationCategory<OpID::LoadOp> {
    using type = MemoryCategory;
    static constexpr uint64_t category_id = TraitID::Memory;
};

template<>
struct OperationCategory<OpID::StoreOp> {
    using type = MemoryCategory;
    static constexpr uint64_t category_id = TraitID::Memory;
};

// Elegant category checking via type traits
template<uint64_t op_id>
using OperationCategory_t = typename OperationCategory<op_id>::type;

template<uint64_t op_id>
constexpr bool isArithmetic() {
    return std::is_same_v<OperationCategory_t<op_id>, ArithmeticCategory>;
}

template<uint64_t op_id>
constexpr bool isMemory() {
    return std::is_same_v<OperationCategory_t<op_id>, MemoryCategory>;
}

template<uint64_t op_id>
constexpr bool isControl() {
    return std::is_same_v<OperationCategory_t<op_id>, ControlCategory>;
}

// Backward compatibility wrapper (can be removed if desired)
template<uint64_t op_id>
struct OpCategory {
    static constexpr bool isArithmetic() { return ::mlir_crtp::isArithmetic<op_id>(); }
    static constexpr bool isMemory() { return ::mlir_crtp::isMemory<op_id>(); }
    static constexpr bool isControl() { return ::mlir_crtp::isControl<op_id>(); }
};

} // namespace mlir_crtp 