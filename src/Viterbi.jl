module Viterbi

"""
   Viterbi.decode(seq, start, stop, K, E)

Infer the best sequence of states using the Viterbi algorithm.
"""
decode(seq, start::Int, stop::Int, K::AbstractMatrix, E::AbstractMatrix) =
    decode(seq, 1:length(seq), start, stop, (k1, k2) -> K[k1, k2], (k, t) -> E[k, t])

function decode(seq, states, start, stop, ts, es)
    m = length(seq)
    trellis = Trellis(length(seq), length(states))
    # initial step: previous state can only be `start`
    for k in states
        score = ts(start, k) + es(k, first(seq))
        update!(trellis, 1, start, k, score)
    end
    # rest of the forward pass
    for i in 2:m
        obs = seq[i]
        for k in states
            k′, score = argmax_k′(obs, i, k, states, trellis, ts, es)
            update!(trellis, i, k′, k, score)
        end
    end
    # argmax looking back from the final state
    final_scores = [ts(k, stop) + score_upto(trellis, m, k) for k in states]
    final_score, idx = findmax(final_scores)
    return follow_backpointers(trellis, states[idx]), final_score
end

function argmax_k′(obs, t, k, states, trellis, ts, es)
    escore = es(k, obs)
    scores = [ts(k′, k) + escore + score_upto(trellis, t-1, k′) for k′ in states]
    best_score, index = findmax(scores)
    return states[index], best_score
end

# matrix of (backpointer, score) tuples
# each row:    state
# each column: observation in sequence
struct Trellis{W<:Real}
    trellis::Matrix{Tuple{Int,W}}
end

Trellis(sequence_length::Int, nstates::Int) =
    Trellis(Float64, sequence_length, nstates)

Trellis(W::Type{<:Real}, sequence_length::Int, nstates::Int) =
    Trellis{W}(fill((0, zero(W)), (nstates, sequence_length)))

# best previous state, looking back from `state` at timestep `t`
backpointer(trellis::Trellis, t, state) = first(trellis.trellis[state, t])

score_upto(trellis::Trellis, t, state) = last(trellis.trellis[state, t])

function follow_backpointers(trellis::Trellis, final_state)
    N = size(trellis.trellis, 1)
    bps = [final_state]
    state = final_state
    for t in N:-1:2
        state = backpointer(trellis, t, state)
        push!(bps, state)
    end
    return reverse(bps)
end

# update the trellis at time t with a k′ -> k transition
function update!(trellis::Trellis, t, k′, k, score)
    trellis.trellis[k, t] = (k′, score)
    return trellis
end

end # module
