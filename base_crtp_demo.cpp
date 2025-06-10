#include <iostream>
#include <string>
#include <type_traits>

// =================== Part 1: Framework Base ===================
// Simulated MLIR Type/Value
struct Type { std::string name; };
struct Value { Type type; std::string name; };

template<typename Derived>
class OpBase {
private:
    Derived* derived() { return static_cast<Derived*>(this); }

public:
    // Unified interface methods - always delegate to derived
    auto getInput() { return derived()->default_getInput(); }
    std::string getOperationName() { return derived()->default_getOperationName(); }
    bool verify() { return derived()->default_verify(); }
    void print() { derived()->default_print(); }
    
    // Default implementations - derived classes can selectively override
    auto default_getInput() {
        std::cout << "Using default input access" << std::endl;
        return Value{Type{"unknown"}, "default_input"}; 
    }
    
    std::string default_getOperationName() {
        std::cout << "Using default operation name" << std::endl;
        return "generic_op";
    }
    
    bool default_verify() {
        std::cout << "Using default verification" << std::endl;
        return true;
    }
    
    void default_print() {
        std::cout << "Op: " << getOperationName() << " (default print)" << std::endl;
    }
};

// =================== Part 2: Basic Usage Examples ===================

// Example 1: Use all default implementations
class SimpleOp : public OpBase<SimpleOp> {
    // No overrides - fully relies on framework defaults
public:
    void demo() {
        std::cout << "\n=== SimpleOp (all defaults) ===" << std::endl;
        getInput();
        std::cout << "Operation: " << getOperationName() << std::endl;
        std::cout << "Valid: " << verify() << std::endl;
        print();
    }
};

// Example 2: Selective override - only custom verification
class IdentityOp : public OpBase<IdentityOp> {
public:
    // Only override verification logic, others use defaults
    bool default_verify() {
        std::cout << "IdentityOp: Custom verification logic" << std::endl;
        return true; // Identity operation is always valid
    }
    
    void demo() {
        std::cout << "\n=== IdentityOp (custom verify) ===" << std::endl;
        getInput();
        std::cout << "Operation: " << getOperationName() << std::endl;
        std::cout << "Valid: " << verify() << std::endl;
        print();
    }
};

// Example 3: Multiple overrides - custom name and print
class ComplexOp : public OpBase<ComplexOp> {
public:
    // Override multiple methods
    std::string default_getOperationName() {
        std::cout << "ComplexOp: Custom operation name" << std::endl;
        return "complex_operation";
    }
    
    void default_print() {
        std::cout << "ComplexOp: Custom print - " << getOperationName() 
                  << " with special formatting" << std::endl;
    }
    
    void demo() {
        std::cout << "\n=== ComplexOp (custom name + print) ===" << std::endl;
        getInput();
        std::cout << "Operation: " << getOperationName() << std::endl;
        std::cout << "Valid: " << verify() << std::endl;
        print();
    }
};

// =================== Part 3: Advanced Features ===================

// Binary operation kinds
enum class BinaryOpKind { Add, Sub, Mul };

// Template specialization - more elegant than if constexpr
template<BinaryOpKind Kind> struct BinaryOpTraits;
template<> struct BinaryOpTraits<BinaryOpKind::Add> { 
    static constexpr const char* name = "add"; 
    static constexpr bool commutative = true;
};
template<> struct BinaryOpTraits<BinaryOpKind::Sub> { 
    static constexpr const char* name = "sub"; 
    static constexpr bool commutative = false;
};
template<> struct BinaryOpTraits<BinaryOpKind::Mul> { 
    static constexpr const char* name = "mul"; 
    static constexpr bool commutative = true;
};

// Template operation with compile-time polymorphism
template<BinaryOpKind Kind>
class BinaryOp : public OpBase<BinaryOp<Kind>> {
public:
    std::string default_getOperationName() {
        std::cout << "BinaryOp: Template-based operation name" << std::endl;
        return BinaryOpTraits<Kind>::name;
    }
    
    bool default_verify() {
        std::cout << "BinaryOp: Validating " << BinaryOpTraits<Kind>::name 
                  << " (commutative: " << BinaryOpTraits<Kind>::commutative << ")" << std::endl;
        return true;
    }
    
    void demo() {
        std::cout << "\n=== BinaryOp<" << BinaryOpTraits<Kind>::name << "> ===" << std::endl;
        this->getInput();
        std::cout << "Operation: " << this->getOperationName() << std::endl;
        std::cout << "Valid: " << this->verify() << std::endl;
        this->print();
    }
};

int main() {
    std::cout << "======= Basic CRTP Demo: Selective Override Pattern =======" << std::endl;
    
    // Basic usage examples
    SimpleOp simple;
    simple.demo();
    
    IdentityOp identity;
    identity.demo();
    
    ComplexOp complex;
    complex.demo();
    
    // Template-based operations
    BinaryOp<BinaryOpKind::Add> addOp;
    addOp.demo();
    
    BinaryOp<BinaryOpKind::Sub> subOp;
    subOp.demo();
    
    std::cout << "\n======= Summary =======" << std::endl;
    std::cout << "✓ SimpleOp: Uses framework defaults" << std::endl;
    std::cout << "✓ IdentityOp: Selective verification override" << std::endl;
    std::cout << "✓ ComplexOp: Multiple method overrides" << std::endl;
    std::cout << "✓ BinaryOp<T>: Template-based compile-time polymorphism" << std::endl;
    std::cout << "\nKey advantage: More flexible than TableGen's fixed extension points!" << std::endl;
    
    return 0;
}

// ================== Comparison Summary ==================

/*
TableGen Approach:
------------------
def IdentityOp : Op<"identity"> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let hasVerifier = 1;
  let extraClassDefinition = [{
    LogicalResult verify() {
      return getInput().getType() == getOutput().getType() ? success() : failure();
    }
  }];
}
// Generates ~200 lines of C++ code

CRTP Approach:
--------------
class IdentityOp : public Op<IdentityOp> {
    LogicalResult default_verify() {
        return (getInput() == getOutput()) ? success() : failure();
    }
    // ~15 lines total, direct C++ code
}

Result: Same functionality, 90% less code, 100% more flexibility!
*/ 