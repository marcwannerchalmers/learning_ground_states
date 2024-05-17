
using ITensors
# using ITensorGPU
using HDF5
using LinearAlgebra
#using PastaQ
using Random
# using Plots
using Printf
# using Test
# using StatsBase
using JLD
using ProgressBars
using Debugger
using CodeTracking
using ArgParse
using CSV
using CUDA

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Lx"
            help = "lattice nodes in x direction"
            arg_type = Int
            default = 5
        "--start_id"
            help = "Id of the starting point"
            arg_type = Int
            default = 1
        "--npoints"
            help = "number of correlations computed"
            arg_type = Int
            default = 256
    end

    return parse_args(s)
end

function main()
    # Set to identity to run on CPU
    # gpu = cu
    cpu = ITensors.cpu
    gpu = cu
    ##seed = 1234
    # n = 1
    # n = parse(Int64,ARGS[1])
    # Random.seed!(k * 1234+N)
    # break_on(:error)
    # outputlevel = 0
    parsed_args = parse_commandline()

    Lx = parsed_args["Lx"]
    Ly = 5
    N = Lx * Ly
    nshots = 1000
    Jmin = 0.0
    Jmax = 2.0
    npoints = parsed_args["npoints"]
    start_id = parsed_args["start_id"]
    @printf("Lx=%i, Ly=%i ", Lx, Ly)
    Λ = 1e-10
    noise = 1e-7
    χ = [10, 10, 20, 20, 50, 100, 200, 200, 500, 1000, 1500]
    χ₀ = 10
    ϵ = 1e-4
    nsweeps = 500
    minsweeps = 5
    basepath = "learning_ground_states/final_data/unif_random/"
    path_j = "learning_ground_states/new_data/data_9x5/simulation_9x5_id1_couplings.txt"


    groundstate = nothing
    for k in start_id:(start_id+npoints-1)
        Random.seed!(k * 1234 + N)
        @printf("iteration %i \n", k)
        sites = siteinds("Qubit", N; conserve_qns=true)
        #lattice = squareLattice(Lx, Ly; Yperiodic=false, Xperiodic=false)
        lattice = square_lattice(Lx, Ly; yperiodic=false)

        # TODO: replace J by LDS
        #=
        f = open(path_j)
        J = Array{Float64}(undef, 76)
        i = 0
        for line in readlines(f)
            i += 1       
            J[i] = parse(Float64, line)     
        end
        close(f)
        =#
        

        J = Jmax * rand(length(lattice))

        ampo = AutoMPO()
        for (b, bond) in enumerate(lattice)
            ampo .+= J[b], "X", bond.s1, "X", bond.s2
            ampo .+= J[b], "Y", bond.s1, "Y", bond.s2
            ampo .+= J[b], "Z", bond.s1, "Z", bond.s2
        end


        H = gpu(MPO(ampo, sites))
        #H = MPO(ampo, sites)
        st = [isodd(n) ? "1" : "0" for n = 1:N]
        ψ₀ = gpu(randomMPS(sites, st))
        # ψ₀ = randomMPS(sites, st)


        sweeps = Sweeps(nsweeps)
        maxdim!(sweeps, χ...)
        cutoff!(sweeps, Λ)
        noise!(sweeps, noise)
        @printf("Running dmrg for %i x %i grid\n", Lx, Ly)
        observer =gpu(DMRGObserver(["Z"], sites, energy_tol=ϵ, minsweeps=minsweeps))
        try
            @time E, ψ = dmrg(H, ψ₀, sweeps; observer=observer, outputlevel=1)

            #CUDA.allowscalar(true)
            ψ = LinearAlgebra.normalize(ψ)
            #SvN = entanglemententropy(ψ)
            ## X      = measure(ψ, "X")
            ## Z      = measure(ψ, "Z")
            # XX    = measure(ψ, ("X", "X")) 
            # YY    = measure(ψ, ("Y", "Y")) 
            # ZZ    = measure(ψ, ("Z", "Z")) 

            # this is the same as above, but in new version
            # X = expect(ψ, "X")
            # Y = expect(ψ, "Y")
            # Z = expect(ψ, "Z")
            @printf("Computing correlation matrix\n")
            XX = correlation_matrix(ψ, "X", "X")
            YY = correlation_matrix(ψ, "Y", "Y")
            ZZ = correlation_matrix(ψ, "Z", "Z")

            XX = cpu(XX)
            YY = cpu(YY)
            ZZ = cpu(ZZ)
            
            # st = [isodd(n) ? "1" : "0" for n = 1:N]
            # ψ = randomMPS(BigFloat, sites, st)
            # ψ = cpu(ψ)
            # ψ = dense(ψ)
            # LinearAlgebra.normalize!(ψ)


            # @printf("Sampling\n")
            # samples = getsamples(LinearAlgebra.normalize(ψ), randombases(N, nshots), 1) # nshots different meas. bases, each base with one shot
            """for i in ProgressBar(1:nshots)
                ψₛ = ψ
                ψₛ = gpu(ψₛ)
                v1 = sample!(dense(ψₛ))
            end"""

            @bp
            @printf("storing results\n")
            path = basepath * "simulation_$(Lx)x$(Ly)_id$(k).jld"
            JLD.jldopen(path, "w") do fout
                JLD.write(fout, "J", J)
                JLD.write(fout, "ZZ", ZZ)
                JLD.write(fout, "YY", YY)
                JLD.write(fout, "XX", XX)
                JLD.write(fout, "E", E)
                #JLD.write(fout, "SvN", SvN)
                # JLD.write(fout, "samples", samples)
            end

            # fout = h5open(basepath * "simulation_$(Lx)x$(Ly)_id$(k)_gs.jld", "w")
            # write(fout, "psi", ψ)
            # close(fout)
        catch e
            showerror(stdout, e)
            @printf("Sampling failed for sample %i\n", k)
            # saving error states
            path = basepath * "failed_simulation_$(Lx)x$(Ly)_id$(k).jld"
            JLD.jldopen(path, "w") do fout
                JLD.write(fout, "J", J)
                # JLD.write(fout, "ZZ", ZZ)
                # JLD.write(fout, "YY", YY)
                # JLD.write(fout, "XX", XX)
                # JLD.write(fout, "E", E)
                #JLD.write(fout, "SvN", SvN)
            end

            # fout = h5open(basepath * "failed_simulation_$(Lx)x$(Ly)_id$(k)_gs.jld", "w")
            # write(fout, "psi", ψ)
            # close(fout)
        end
        println("")
        # println("samples:")
        # display(samples)
        # println("psi:")
        # display(ψ)
        # println("XX:")
        # display(XX)
        # groundstate = ψ

    end
    return groundstate
end

# print(@code_string main())

if "" != PROGRAM_FILE && realpath(@__FILE__) == realpath(PROGRAM_FILE)
    main()
end

