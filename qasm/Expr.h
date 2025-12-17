/**
 * @file Expr.h
 * @version 1.0
 *
 * @section DESCRIPTION
 *
 * Classes for the qasm parser and interpreter, expressions part.
 *
 * It's supposed to support only open qasm 2.0.
 */
#pragma once

#ifndef _EXPR_H_
#define _EXPR_H_

#ifdef DEBUG
#define BOOST_SPIRIT_QI_DEBUG
#endif  // DEBUG

#define _USE_MATH_DEFINES
#include <math.h>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/io.hpp>
#include <boost/phoenix.hpp>
#include <boost/phoenix/object.hpp>
#include <boost/spirit/include/qi.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace qasm {
namespace qi = boost::spirit::qi;
namespace ascii = boost::spirit::ascii;
namespace phx = boost::phoenix;

class AbstractSyntaxTree {
 public:
  virtual ~AbstractSyntaxTree() = default;
  virtual double Eval() const { return 0; }
  virtual double Eval(
      const std::unordered_map<std::string, double> &variables) const {
    return 0;
  }

 protected:
  AbstractSyntaxTree() = default;
  AbstractSyntaxTree(const AbstractSyntaxTree &) = default;
  AbstractSyntaxTree(AbstractSyntaxTree &&) = default;
  AbstractSyntaxTree &operator=(const AbstractSyntaxTree &) = default;
  AbstractSyntaxTree &operator=(AbstractSyntaxTree &&) = default;
};

typedef std::shared_ptr<AbstractSyntaxTree> AbstractSyntaxTreePtr;

template <typename Expr>
static AbstractSyntaxTreePtr Clone(Expr const &t) {
  return std::make_shared<Expr>(t);
}

// for expressions (to be evaluated, typically those are values for parameters
// for gates)

class Constant : public AbstractSyntaxTree {
 public:
  Constant(double value = 0) : value(value) {}
  Constant(int value) : value(value) {}

  Constant &operator=(int value) {
    this->value = value;
    return *this;
  }
  Constant &operator=(double value) {
    this->value = value;
    return *this;
  }

  double Eval() const override { return value; }
  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    return value;
  }

 private:
  double value;
};

struct MakeConstantExpression {
  template <typename>
  struct result {
    typedef Constant type;
  };

  template <typename C>
  Constant operator()(C op) const {
    return Constant(op);
  }
};

phx::function<MakeConstantExpression> MakeConstant;

class Variable : public AbstractSyntaxTree {
 public:
  Variable(const std::string &value = "") : value(value) {}

  Variable &operator=(int value) {
    this->value = value;
    return *this;
  }

  double Eval() const override { return 0; }
  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    auto it = variables.find(value);
    if (it != variables.end())
      return it->second;
    else
      throw std::invalid_argument("Variable not found: " + value);

    return 0;
  }

 private:
  std::string value;
};

struct MakeVariableExpression {
  template <typename>
  struct result {
    typedef Variable type;
  };

  template <typename V>
  Variable operator()(V v) const {
    return Variable(v);
  }
};

phx::function<MakeVariableExpression> MakeVariable;

class BinaryOperator : public AbstractSyntaxTree {
 public:
  template <typename L, typename R>
  BinaryOperator(char op, const L &left, const R &right)
      : op(op), left(Clone(left)), right(Clone(right)) {}

  double Eval() const override {
    switch (op) {
      case '+':
        return left->Eval() + right->Eval();
      case '-':
        return left->Eval() - right->Eval();
      case '*':
        return left->Eval() * right->Eval();
      case '/':
        return left->Eval() / right->Eval();
      case '^':
        return pow(left->Eval(), right->Eval());
      default:
        throw std::invalid_argument("Unknown binary operator");
    }

    return 0;
  }

  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    switch (op) {
      case '+':
        return left->Eval(variables) + right->Eval(variables);
      case '-':
        return left->Eval(variables) - right->Eval(variables);
      case '*':
        return left->Eval(variables) * right->Eval(variables);
      case '/':
        return left->Eval(variables) / right->Eval(variables);
      case '^':
        return pow(left->Eval(variables), right->Eval(variables));
      default:
        throw std::invalid_argument("Unknown binary operator");
    }

    return 0;
  }

 private:
  char op;
  AbstractSyntaxTreePtr left, right;
};

struct MakeBinaryExpression {
  template <typename, typename, typename>
  struct result {
    typedef BinaryOperator type;
  };

  template <typename C, typename L, typename R>
  BinaryOperator operator()(C op, const L &lhs, const R &rhs) const {
    return BinaryOperator(op, lhs, rhs);
  }
};

phx::function<MakeBinaryExpression> MakeBinary;

class UnaryOperator : public AbstractSyntaxTree {
 public:
  UnaryOperator() : op('+') {}

  template <typename R>
  UnaryOperator(char op, const R &right) : op(op), right(Clone(right)) {}

  double Eval() const override {
    switch (op) {
      case '+':
        return right->Eval();
      case '-':
        return -right->Eval();
      default:
        throw std::invalid_argument("Unknown unary operator");
    }

    return 0;
  }

  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    switch (op) {
      case '+':
        return right->Eval(variables);
      case '-':
        return -right->Eval(variables);
      default:
        throw std::invalid_argument("Unknown unary operator");
    }
    return 0;
  }

 private:
  char op;
  AbstractSyntaxTreePtr right;
};

struct MakeUnaryExpression {
  template <typename, typename>
  struct result {
    typedef UnaryOperator type;
  };

  template <typename C, typename R>
  UnaryOperator operator()(C op, const R &rhs) const {
    return UnaryOperator(op, rhs);
  }
};

phx::function<MakeUnaryExpression> MakeUnary;

class Function : public AbstractSyntaxTree {
 public:
  template <typename F>
  Function(const std::string &func, const F &param)
      : func(func), param(Clone(param)) {}

  double Eval() const override {
    if (func == "sin")
      return sin(param->Eval());
    else if (func == "cos")
      return cos(param->Eval());
    else if (func == "tan")
      return tan(param->Eval());
    else if (func == "exp")
      return exp(param->Eval());
    else if (func == "ln")
      return log(param->Eval());
    else if (func == "sqrt")
      return sqrt(param->Eval());

    throw std::invalid_argument("Unknown function");

    return 0;
  }

  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    if (func == "sin")
      return sin(param->Eval(variables));
    else if (func == "cos")
      return cos(param->Eval(variables));
    else if (func == "tan")
      return tan(param->Eval(variables));
    else if (func == "exp")
      return exp(param->Eval(variables));
    else if (func == "ln")
      return log(param->Eval(variables));
    else if (func == "sqrt")
      return sqrt(param->Eval(variables));

    throw std::invalid_argument("Unknown function");

    return 0;
  }

 private:
  std::string func;
  AbstractSyntaxTreePtr param;
};

struct MakeFunctionExpression {
  template <typename, typename, typename>
  struct result {
    typedef Function type;
  };

  template <typename Params>
  Function operator()(const std::string &funcName, const Params &params) const {
    return Function(funcName, params);
  }
};

phx::function<MakeFunctionExpression> MakeFunction;

class Expression : public AbstractSyntaxTree {
 public:
  Expression() {}
  ~Expression() override {}

  template <typename E>
  Expression(E const &e) : expr(Clone(e)) {}

  double Eval() const override { return expr->Eval(); }

  double Eval(
      const std::unordered_map<std::string, double> &variables) const override {
    return expr->Eval(variables);
  }

  friend AbstractSyntaxTreePtr Clone(Expression const &e) { return e.expr; }

 private:
  AbstractSyntaxTreePtr expr;
};
}  // namespace qasm

#endif
