#include "../advanced_bind_from_value_to_type.hpp"
#include <iostream>
#include <cstdio>

using namespace mlir_crtp;

// =================== Simple Demo ===================

// Type registration and minimal trait binding for testing
namespace mlir_crtp {
    // Type registration
    template <>
    struct RegisteredType<TypeID::IntegerType32> : public Type2Type<int32_t> {};
    
    template <>
    struct RegisteredType<TypeID::FloatType32> : public Type2Type<float> {};
    
    // Minimal trait binding for SimpleOp (AddI operation)
    template <>
    struct OpTraitList<OpID::AddI> : public Type2Type<TypeList<int>> {};  // Use int as placeholder trait
}

// Simple operation without complex traits
class SimpleOp : public ValueBoundOp<OpID::AddI, SimpleOp> {
    int value_;
    
public:
    SimpleOp() : value_(42) {}
    SimpleOp(int value) : value_(value) {}
    
    bool default_verify() { 
        printf("Verifying SimpleOp with value: %d\n", value_);
        return true; 
    }
    
    void default_print() { 
        printf("SimpleOp: value = %d\n", value_);
    }
    
    int default_getInput() { return value_; }
};

// Test TypeList operations
void testTypeListOperations() {
    printf("\n=== TypeList Operations Test ===\n");
    
    // Test type extraction
    using TestList = TypeList<int, float, double, char>;
    using SecondType = GetNthTypeInTypeList_t<1, TestList>; // Should be float
    
    static_assert(std::is_same_v<SecondType, float>, "Type extraction failed!");
    printf("✓ Type extraction works: Got float as expected\n");
    
    // Test type replacement  
    using ModifiedList = ReplaceNthTypeInList_t<2, bool, TestList>; // Replace double with bool
    using NewThirdType = GetNthTypeInTypeList_t<2, ModifiedList>; // Should be bool
    
    static_assert(std::is_same_v<NewThirdType, bool>, "Type replacement failed!");
    printf("✓ Type replacement works: Replaced double with bool\n");
    
    printf("TypeList operations: SUCCESS\n");
}

// Test dispatch functionality
void testDispatch() {
    printf("\n=== Dispatch Test ===\n");
    
    // Test compile-time dispatch
    constexpr auto arith_result = dispatch_operation<OpID::AddI>();
    printf("AddI dispatch result: %s\n", arith_result);
    
    constexpr auto mem_result = dispatch_operation<OpID::LoadOp>();
    printf("LoadOp dispatch result: %s\n", mem_result);
    
    printf("Dispatch test: SUCCESS\n");
}

// Test operation categories
void testCategories() {
    printf("\n=== Category Test ===\n");
    
    constexpr bool is_add_arithmetic = OpCategory<OpID::AddI>::isArithmetic();
    constexpr bool is_load_memory = OpCategory<OpID::LoadOp>::isMemory();
    
    printf("AddI is arithmetic: %s\n", is_add_arithmetic ? "YES" : "NO");
    printf("LoadOp is memory: %s\n", is_load_memory ? "YES" : "NO");
    
    static_assert(is_add_arithmetic, "AddI should be arithmetic");
    static_assert(is_load_memory, "LoadOp should be memory");
    
    printf("Category test: SUCCESS\n");
}

// Test symbol optimization
void testSymbolOptimization() {
    printf("\n=== Symbol Optimization Test ===\n");
    
    using OptimizedAddI = OptimizedOp<OpID::AddI, TypeID::IntegerType32, TypeID::IntegerType32>;
    
    printf("Optimized symbol name: %s\n", OptimizedAddI::getSymbolName());
    printf("This is much shorter than traditional MLIR symbols!\n");
    
    printf("Symbol optimization: SUCCESS\n");
}

int main() {
    printf("=== Advanced Value-to-Type Binding Demo ===\n");
    printf("Author: Shen Xiaolong\n");
    printf("Repository: https://github.com/shenxiaolong-code/mlir-crtp-proposal\n\n");
    
    // Test basic operation
    printf("=== Basic Operation Test ===\n");
    SimpleOp op(123);
    op.verify();
    op.print();
    printf("Operation ID: 0x%lx\n", SimpleOp::operation_id);
    printf("Basic operation: SUCCESS\n");
    
    // Run all tests
    testTypeListOperations();
    testDispatch();
    testCategories();
    testSymbolOptimization();
    
    printf("\n=== All Tests Completed Successfully! ===\n");
    printf("Value-to-Type binding system is working correctly.\n");
    
    return 0;
} 