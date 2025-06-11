# MLIR CRTP Alternative: Modern C++ Approach to Operation Definition

A comprehensive proposal for using **CRTP (Curiously Recurring Template Pattern)** as an alternative to TableGen for defining MLIR operations, offering enhanced flexibility, lower learning barrier, and better development experience.

[🇨🇳 中文版本](./README_cn.md)

MLIR Community Discussion : [rfc-modern-c-alternative-to-tablegen-for-mlir-operation-definition](https://discourse.llvm.org/t/rfc-modern-c-alternative-to-tablegen-for-mlir-operation-definition/86800)
 
## ⚠️ Error/Exception Handling

**Error/Exception Handling**: This project's demonstration code focuses on showcasing core technical concepts and essential functionality. For clarity and to highlight key points, detailed error handling (boundary checks, exception handling, etc.) has been omitted. Please add appropriate error handling mechanisms in production environments.

## 🎯 Core Concept

Instead of generating C++ from DSL, write C++ directly with modern patterns:

```cpp
// Traditional TableGen approach
def AddOp : Op<"add", [Commutative]> {
  let hasFolder = 1;  // Limited to predefined extension pointsl
}
// Generates ~200 lines of C++ code

// CRTP approach  
template<> struct trait_binding<AddOp> : Type2Type<ArithmeticTrait<AddOp>> {};

class AddOp : public Op<AddOp> {
    // Automatically inherits ArithmeticTrait capabilities
    // Can override any base class method
    // Can add custom methods
};
// ~15 lines total, direct C++ code
```

## 📊 Key Advantages

| Aspect | TableGen | CRTP Approach |
|--------|----------|---------------|
| **Learning Curve** | New DSL syntax | Standard C++ patterns |
| **Customization** | Fixed extension points | Any method overridable |
| **Code Generation** | 200+ lines per op | 0 lines generated |
| **IDE Support** | Limited | Full C++ tooling |
| **Debugging** | Generated code | Your actual code |
| **Template Support** | Basic | Full C++ templates |
| **Performance** | Zero overhead | Zero overhead |

## 🚀 Quick Start

### Clone and Run Demos

```bash
git clone https://github.com/shenxiaolong-code/mlir-crtp-proposal
cd mlir-crtp-proposal
cd test

# Method 1: Use Makefile (recommended)
make all          # Build all demos
make test         # Run advanced value-binding demo
make help         # Show all available targets

# Method 2: Manual compilation  
g++ -std=c++17 base_crtp_demo.cpp -o base_demo && ./base_demo
g++ -std=c++17 enhanced_crtp_trait_bind_demo.cpp -o enhanced_demo && ./enhanced_demo
g++ -std=c++17 advanced_bind_demo.cpp -o advanced_demo && ./advanced_demo
```



### Expected Output
The demos showcase:
- **Selective Override**: Implement only what you need
- **Trait Binding**: Declarative trait application
- **Compile-time Safety**: Full type checking and optimization

## 🏗️ Architecture Overview

### 1. Selective Override Pattern
```cpp
template<typename Derived>
class OpBase {
public:
    // Unified interface - always delegates to derived
    auto getInput() { return derived()->default_getInput(); }
    bool verify() { return derived()->default_verify(); }
    
    // Default implementations - selectively overridable
    auto default_getInput() { /* framework default */ }
    bool default_verify() { return true; }
    
private:
    Derived* derived() { return static_cast<Derived*>(this); }
};
```

### 2. Trait Binding System
```cpp
// Framework provides intelligent defaults
template<typename T>
struct trait_binding : Type2Type<DefaultTrait<T>> {};

// Users declaratively specify replacements
template<> struct trait_binding<AddOp> : Type2Type<ArithmeticTrait<AddOp>> {};
template<> struct trait_binding<LoadOp> : Type2Type<MemoryTrait<LoadOp>> {};

// Operations automatically inherit appropriate traits
class AddOp : public Op<AddOp> {
    // Gets ArithmeticTrait capabilities automatically
};
```

## 📁 Project Structure

```
mlir-crtp-proposal/
├── README.md                               # English version (this file)
├── README_cn.md                            # Chinese version
├── LICENSE                                 # MIT License
├── MLIR_CRTP_RFC_Proposal.md              # Complete technical RFC (English)
├── MLIR_CRTP_RFC_Proposal_cn.md           # Complete technical RFC (Chinese)
├── base_crtp_demo.cpp                      # Basic CRTP patterns
├── base_crtp_demo.md                       # Basic demo guide (English)
├── base_crtp_demo_cn.md                    # Basic demo guide (Chinese)
├── enhanced_crtp_trait_bind_demo.cpp       # Full trait_binding system
├── enhanced_crtp_trait_bind_demo.md        # Enhanced demo guide (English)
└── enhanced_crtp_trait_bind_demo_cn.md     # Enhanced demo guide (Chinese)
```

## 📚 Documentation

### For Developers
- [`base_crtp_demo.md`](./base_crtp_demo.md) - Introduction to CRTP patterns
- [`enhanced_crtp_trait_bind_demo.md`](./enhanced_crtp_trait_bind_demo.md) - Complete trait_binding system
- [`advanced_bind_from_value_to_type.md`](./advanced_bind_from_value_to_type.md) - 🔥 **Industrial type-to-value-binding techniques**
- [`MLIR_CRTP_RFC_Proposal.md`](./MLIR_CRTP_RFC_Proposal.md) - Comprehensive technical proposal

### Key Features Demonstrated
- **Lower Learning Barrier**: Based on standard C++ patterns, no need to learn additional TableGen DSL
- **Complete Flexibility**: Override any method, not just extension points  
- **Non-invasive Design**: Framework unchanged, users add specializations
- **Type Safety**: Compile-time trait detection and verification
- **Modern C++**: Template specialization, constexpr, concepts
- **🚀 Advanced Value Binding**: Symbol optimization techniques (proof-of-concept stage)
- **🔥 Symbol Table Reduction**: Theoretical analysis shows significant reduction in MLIR template bloat (requires validation)
- **🎨 Advanced Type Computation**: Using [MiniMPL](https://github.com/shenxiaolong-code/MiniMPL) template metaprogramming techniques

### Performance Claims Clarification
The performance advantages in this project are primarily based on:
- **Theoretical Analysis**: Based on C++ template mechanism's compile-time optimization characteristics
- **Proof of Concept**: Test results from small-scale demonstration code
- **Requires Validation**: Large-scale real project performance remains to be verified

Benchmarking in actual projects is recommended to verify specific performance benefits.

## 🔬 Technical Highlights

### Advanced Type Computation Framework
Our approach leverages template metaprogramming techniques from the [MiniMPL TypeList library](https://github.com/shenxiaolong-code/MiniMPL/blob/master/sources/MiniMPL/include/MiniMPL/typeList_cpp11.hpp), enabling:

- **Industrial-Grade Type Manipulation**: Advanced algorithms for compile-time type operations
- **Dynamic Dialect Generation**: Automatic creation of operation variants through type transformations
- **Compile-Time Optimization**: Zero runtime overhead with sophisticated compile-time dispatch

### Template Specialization Elegance
Instead of complex conditional logic:
```cpp
// Clean traits hierarchy
template<BinaryOpKind Kind> struct BinaryOpTraits;
template<> struct BinaryOpTraits<BinaryOpKind::Add> { 
    static constexpr const char* name = "add"; 
    static constexpr bool commutative = true;
};
template<> struct BinaryOpTraits<BinaryOpKind::Sub> { 
    static constexpr const char* name = "sub"; 
    static constexpr bool commutative = false;
};
```

### Compile-time Trait Detection
```cpp
template<typename OpType>
void analyzeOperation(OpType& op) {
    if constexpr (OpType::template hasTrait<ArithmeticTrait<OpType>>()) {
        op.getTrait()->fold();  // Only compiled for arithmetic ops
    }
    
    if constexpr (OpType::template hasTrait<MemoryTrait<OpType>>()) {
        op.getTrait()->getMemorySlots();  // Only compiled for memory ops
    }
}
```

## 🎯 Why This Matters

1. **Zero Code Generation**: Direct C++ means no intermediate generated code
2. **Better Development Experience**: Full IDE support, direct debugging, standard refactoring
3. **Modern C++ Features**: Template specialization, constexpr, concepts, etc.
4. **Gradual Migration**: Can coexist with TableGen during transition
5. **Community Familiarity**: Every MLIR developer already knows C++

## 🤝 Contributing

This project welcomes contributions! Whether you're interested in:
- Extending the demo implementations
- Improving documentation
- Testing with real MLIR dialects
- Performance benchmarking
- Migration tooling development

Please feel free to submit issues, pull requests, or start discussions.

## 📬 Community Discussion

This proposal is part of an ongoing discussion in the MLIR community. For detailed technical analysis and community feedback:

- **LLVM Discourse**: [RFC Discussion Thread](https://discourse.llvm.org/c/mlir/31)
- **GitHub Issues**: [Technical questions and feature requests](https://github.com/shenxiaolong-code/mlir-crtp-proposal/issues)
- **Pull Requests**: [Code improvements and extensions](https://github.com/shenxiaolong-code/mlir-crtp-proposal/pulls)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLIR Community**: For building an excellent infrastructure
- **LLVM Project**: For providing the foundation
- **Modern C++ Community**: For advancing template metaprogramming techniques

---

**🔗 Links**
- [Complete RFC Document](./MLIR_CRTP_RFC_Proposal.md)
- [Basic Demo Guide](./base_crtp_demo.md)  
- [Enhanced Demo Guide](./enhanced_crtp_trait_bind_demo.md)
- [GitHub Repository](https://github.com/shenxiaolong-code/mlir-crtp-proposal)

 