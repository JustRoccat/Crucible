import pytest
import z3
from src.symbolic.z3_engine import (
    SymbolicSpec,
    UnsafeExpressionError,
    Z3LogicalEngine,
    safe_z3_expr,
)


def test_safe_z3_expr_basic_comparison():
    n = z3.Int("n")
    expr = safe_z3_expr("n > 0", {"n": n})
    s = z3.Solver()
    s.add(expr, n < 5)
    assert s.check() == z3.sat


def test_safe_z3_expr_and_or_not():
    n, m = (z3.Int("n"), z3.Int("m"))
    expr = safe_z3_expr("(n > 0 and n < 10) or not (m == 0)", {"n": n, "m": m})
    s = z3.Solver()
    s.add(expr, n == -5, m == 1)
    assert s.check() == z3.sat


def test_safe_z3_expr_chained_comparison():
    n = z3.Int("n")
    expr = safe_z3_expr("0 < n < 10", {"n": n})
    s = z3.Solver()
    s.add(expr)
    s.add(n == 20)
    assert s.check() == z3.unsat


def test_safe_z3_expr_floordiv_regression():
    n = z3.Int("n")
    expr = safe_z3_expr("n == 7 // 2", {"n": n})
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat
    assert s.model()[n].as_long() == 3


def test_safe_z3_expr_floordiv_with_variable():
    n = z3.Int("n")
    expr = safe_z3_expr("n // 2 == 3", {"n": n})
    s = z3.Solver()
    s.add(expr, n > 0, n < 10)
    assert s.check() == z3.sat
    val = s.model()[n].as_long()
    assert val // 2 == 3


def test_safe_z3_expr_negative_unary():
    n = z3.Int("n")
    expr = safe_z3_expr("n == -5", {"n": n})
    s = z3.Solver()
    s.add(expr)
    assert s.check() == z3.sat
    assert s.model()[n].as_long() == -5


@pytest.mark.parametrize(
    "expr_str",
    [
        "__import__('os').system('echo hi')",
        "n.bit_length()",
        "(lambda: 1)()",
        "[x for x in range(3)]",
        "n if n > 0 else -n",
        "open('file.txt')",
    ],
)
def test_safe_z3_expr_rejects_unsafe_constructs(expr_str):
    n = z3.Int("n")
    with pytest.raises(UnsafeExpressionError):
        safe_z3_expr(expr_str, {"n": n})


def test_safe_z3_expr_rejects_unknown_variable():
    n = z3.Int("n")
    with pytest.raises(UnsafeExpressionError):
        safe_z3_expr("m > 0", {"n": n})


def test_safe_z3_expr_rejects_syntax_error():
    n = z3.Int("n")
    with pytest.raises(UnsafeExpressionError):
        safe_z3_expr("n >", {"n": n})


def test_check_spec_sat_returns_model_values():
    spec = SymbolicSpec(var_names=["n", "m"], constraints=["n > 0", "n < m", "m < 10"])
    result = Z3LogicalEngine.check_spec(spec)
    assert result.status == "sat"
    assert 0 < result.model_values["n"] < result.model_values["m"] < 10


def test_check_spec_unsat_names_offending_constraints():
    spec = SymbolicSpec(var_names=["n"], constraints=["n > 10", "n < 5"])
    result = Z3LogicalEngine.check_spec(spec)
    assert result.status == "unsat"
    assert set(result.unsat_core) <= {"n > 10", "n < 5"}
    assert len(result.unsat_core) > 0


def test_check_spec_returns_error_status_for_unsafe_constraint_instead_of_raising():
    spec = SymbolicSpec(var_names=["n"], constraints=["__import__('os')"])
    result = Z3LogicalEngine.check_spec(spec)
    assert result.status == "error"
    assert result.model_values is None


def test_solve_bounds_check_sat_and_unsat():
    assert Z3LogicalEngine.solve_bounds_check("x", 0, 10).startswith("SAT")
    assert (
        Z3LogicalEngine.solve_bounds_check("x", 10, 0) == "UNSAT: sprzecznosc logiczna"
    )
