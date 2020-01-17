using Viterbi
using Test

@testset "Viterbi.jl" begin
    @test 1 == 1

    @testset "Eisenstein Intro to NLP Figure 7.1" begin
        sentence = ["they", "can", "fish"]
        tagset = ["N", "V"]
        w2i = Dict("they" => 1, "can" => 2, "fish" => 3, "<S>" => 4)
        t2i = Dict("N" => 1, "V" => 2, "<S>" => 3)
        emission_weights = [-2  -3 -3;
                            -10 -1 -3;
                            -Inf -Inf -Inf]
        transition_weights = [-3 -1 -1;
                              -1 -3 -1;
                              -1 -2 -Inf]
        ts(k′::Int, k::Int) = transition_weights[k′, k]
        es(k::Int,  t::Int) = emission_weights[k, t]
        ts(k′::String, k::String) = ts(t2i[k′], t2i[k])
        es(k′::String, t::String) = es(t2i[k′], w2i[t])

        @assert [es(tag, word) for tag in tagset, word in sentence] == emission_weights[[1,2],:]

        @test Viterbi.decode([1,2,3],[1,2,3], 3, 3, ts, es) == ([1,2,1], -10)
    end
end
