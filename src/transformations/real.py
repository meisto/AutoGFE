# ======================================================================
# Author: Tobias Meisel (meisto)
# Creation Date: Wed 22 Dec 2021 02:56:53 PM CET
# Description: -
# ======================================================================
import numpy as np

from src.env.feature import Transformation, register_transformation_lambda
from src.transformations.transformation import sanity_check

# Square function
square = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(["POSITIVE"]),
    arity = 1,
    name = "square"
)
register_transformation_lambda(square, lambda x: x[0] ** 2)

# Logarithmic function
log = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(["POSITIVE", "NONZERO"]),
    produced_tags = frozenset(),
    arity = 1,
    name = "log"
)
register_transformation_lambda(log, lambda x: np.log(x[0]))

# Absolute function
absolute = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(["POSITIVE"]),
    arity = 1,
    name = "absolute"
)
register_transformation_lambda(absolute, lambda x: np.abs(x[0]))

# square_root function
square_root = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(["POSITIVE"]),
    produced_tags = frozenset(),
    arity = 1,
    name = "square_root"
)
register_transformation_lambda(square_root, lambda x: np.sqrt(x[0]))

# round function
round_fun = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 1,
    name = "round",
)
register_transformation_lambda(round_fun, lambda x: np.round(x[0]))

# tanh function
tanh = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 1,
    name = "tanh",
)
register_transformation_lambda(tanh, lambda x: np.tanh(x[0]))

# sigmoid function
sigmoid = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 1,
    name = "sigmoid",
)
register_transformation_lambda(sigmoid, lambda x: 1/(np.exp(-x[0]) + 1))

# function
normalization = Transformation(
    required_datatypes = ("REAL",),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 1,
    name = "normalization",
)
register_transformation_lambda(normalization, \
    lambda x: (x[0] - x[0].mean()) / (x[0].std() + 10e-4))

# # function
# ... = Transformation(
#     required_datatypes = ("REAL",),
#     produced_datatype = "REAL",
#     required_tags = frozenset(),
#     produced_tags = frozenset(),
#     arity = 1,
#     name = "...",
# )
# register_transformation_lambda(..., lambda x: np.round(x[0]))
# 
# # function
# ... = Transformation(
#     required_datatypes = ("REAL",),
#     produced_datatype = "REAL",
#     required_tags = frozenset(),
#     produced_tags = frozenset(),
#     arity = 1,
#     name = "...",
# )
# register_transformation_lambda(..., lambda x: np.round(x[0]))
# 
# # function
# ... = Transformation(
#     required_datatypes = ("REAL",),
#     produced_datatype = "REAL",
#     required_tags = frozenset(),
#     produced_tags = frozenset(),
#     arity = 1,
#     name = "...",
# )
# register_transformation_lambda(..., lambda x: np.round(x[0]))



## Binary

# Multiplication
multiplication = Transformation(
    required_datatypes = ("REAL", "REAL"),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 2,
    name = "multiplication"
)
register_transformation_lambda(multiplication, lambda x: x[0] * x[1])

# Division
division = Transformation(
    required_datatypes = ("REAL", "REAL"),
    produced_datatype = "REAL",
    required_tags = frozenset("NONZERO"),
    produced_tags = frozenset(),
    arity = 2,
    name = "division",
)
register_transformation_lambda(division, lambda x: np.round(x[0]))

# Addition
addition= Transformation(
    required_datatypes = ("REAL", "REAL"),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 2,
    name = "addition",
)
register_transformation_lambda(addition, lambda x: x[0] + x[1])

# Subtraction
subtraction = Transformation(
    required_datatypes = ("REAL", "REAL"),
    produced_datatype = "REAL",
    required_tags = frozenset(),
    produced_tags = frozenset(),
    arity = 2,
    name = "subtraction",
)
register_transformation_lambda(subtraction, lambda x: x[1] - x[0])

all_transformations = frozenset([
    # Unary
    square,
    log,
    absolute,
    square_root,
    #frequency,
    round_fun,
    tanh,
    #sigmoid,
    #isotonic regression,
    #zscore,
    normalization,

    # Binary
    multiplication,
    addition,
    subtraction,
    division,
])
sanity_check(all_transformations)
