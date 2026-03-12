## Standalone Enzyme MWE: GC-pointer phi nodes in copy! via Dict iteration
##
## Background: SpeedyWeather's Base.copy!(::PrognosticVariables, ...) iterates
## over Dict-like fields using `for (key, value) in pairs(...)`. On GitHub
## Actions (ubuntu-latest, AMD EPYC, Julia 1.10) this occasionally causes
## an Enzyme internal error ("Could not analyze GC behavior" / nodecayed_phis!).
## This MWE attempts to reproduce the pattern without SpeedyWeather.
##
## Run: julia +1.10 mwe_enzyme_standalone.jl

using Enzyme

Enzyme.Compiler.VERBOSE_ERRORS[] = true

# ============================================================
# Minimal struct mimicking SpeedyWeather's PrognosticVariables
# with Dict-based ocean/land fields
# ============================================================

mutable struct FieldContainer
    data::Dict{Symbol, Vector{Float64}}
end

mutable struct StateVec
    ocean::FieldContainer
    land::FieldContainer
    core::Vector{Float64}
end

function Base.copy!(dst::StateVec, src::StateVec)
    for (key, value) in pairs(src.ocean.data)
        dst.ocean.data[key] .= value
    end
    for (key, value) in pairs(src.land.data)
        dst.land.data[key] .= value
    end
    dst.core .= src.core
    return dst
end

function f_copy!(dst, src)
    copy!(dst, src)
    return nothing
end

n = 8
src = StateVec(
    FieldContainer(Dict(:sst => rand(n), :ice => rand(n), :flux => rand(n))),
    FieldContainer(Dict(:soil => rand(n), :temp => rand(n))),
    rand(n),
)
dst = StateVec(
    FieldContainer(Dict(:sst => zeros(n), :ice => zeros(n), :flux => zeros(n))),
    FieldContainer(Dict(:soil => zeros(n), :temp => zeros(n))),
    zeros(n),
)
ddst = StateVec(
    FieldContainer(Dict(:sst => ones(n), :ice => ones(n), :flux => ones(n))),
    FieldContainer(Dict(:soil => ones(n), :temp => ones(n))),
    ones(n),
)
dsrc = StateVec(
    FieldContainer(Dict(:sst => zeros(n), :ice => zeros(n), :flux => zeros(n))),
    FieldContainer(Dict(:soil => zeros(n), :temp => zeros(n))),
    zeros(n),
)

println("Test 1: autodiff through copy! with nested Dict iteration (full StateVec)...")
flush(stdout)
try
    autodiff(Reverse, f_copy!, Const,
        Duplicated(dst, ddst),
        Duplicated(src, dsrc),
    )
    println("SUCCESS")
catch e
    println("FAILED with $(typeof(e)):\n$e")
end

# ============================================================
# Variant: bare Dict loop only (most minimal)
# ============================================================

mutable struct DictState
    data::Dict{Symbol, Vector{Float64}}
end

function g_copy!(dst::DictState, src::DictState)
    for (key, value) in pairs(src.data)
        dst.data[key] .= value
    end
    return nothing
end

src2  = DictState(Dict(:a => rand(n), :b => rand(n), :c => rand(n)))
dst2  = DictState(Dict(:a => zeros(n), :b => zeros(n), :c => zeros(n)))
ddst2 = DictState(Dict(:a => ones(n),  :b => ones(n),  :c => ones(n)))
dsrc2 = DictState(Dict(:a => zeros(n), :b => zeros(n), :c => zeros(n)))

println()
println("Test 2: autodiff through bare Dict-loop copy! (minimal)...")
flush(stdout)
try
    autodiff(Reverse, g_copy!, Const,
        Duplicated(dst2, ddst2),
        Duplicated(src2, dsrc2),
    )
    println("SUCCESS")
catch e
    println("FAILED with $(typeof(e)):\n$e")
end
