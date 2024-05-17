
using ITensors
# using ITensorGPU
using HDF5
using LinearAlgebra
using PastaQ
using Random
# using Plots
using Printf
# using Test
# using StatsBase
using JLD
using ProgressBars
using Debugger
using CodeTracking

using CUDA

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


    Lx = 4
    Ly = 5
    N = Lx * Ly
    nshots = 1000
    Jmin = 0.0
    Jmax = 2.0
    npoints = 10
    start_id = 1002
    @printf("Lx=%i, Ly=%i ", Lx, Ly)
    Λ = 1e-12
    noise = 1e-9
    χ = [10, 10, 20, 20, 50, 100, 200, 200, 500, 1000, 1500]
    χ₀ = 10
    ϵ = 1e-4
    nsweeps = 500
    minsweeps = 5
    basepath = "data_generation/data_test/"

    groundstate = nothing
    for k in start_id:(start_id+npoints-1)
        Random.seed!(k * 1234 + N)
        @printf("iteration %i", k)
        sites = siteinds("Qubit", N; conserve_qns=true)
        #lattice = squareLattice(Lx, Ly; Yperiodic=false, Xperiodic=false)
        lattice = square_lattice(Lx, Ly; yperiodic=false)

        # TODO: replace J by LDS

        J = Jmax * rand(length(lattice))
        
        # println("couplings:")
        # display(J)
        # Define the Heisenberg spin Hamiltonian on this lattice
        # for (b, bond) in enumerate(lattice)
        #     print(bond)
        # end
        ampo = AutoMPO()
        for (b, bond) in enumerate(lattice)
            println(bond.s1, ", ", bond.s2, "\n")
            ampo .+= J[b], "X", bond.s1, "X", bond.s2
            ampo .+= J[b], "Y", bond.s1, "Y", bond.s2
            ampo .+= J[b], "Z", bond.s1, "Z", bond.s2
        end


        H = gpu(MPO(ampo, sites))
        #H = MPO(ampo, sites)
        st = [isodd(n) ? "1" : "0" for n = 1:N]
        ψ₀ = gpu(randomMPS(Float64, sites, st))
        # ψ₀ = randomMPS(sites, st)


        sweeps = Sweeps(nsweeps)
        maxdim!(sweeps, χ...)
        cutoff!(sweeps, Λ)
        noise!(sweeps, noise)
        @printf("Running dmrg for %i x %i grid\n", Lx, Ly)
        observer =gpu(DMRGObserver(["Z"], sites, energy_tol=ϵ, minsweeps=minsweeps))
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
        XX = correlation_matrix(ψ, "X", "X")
        YY = correlation_matrix(ψ, "Y", "Y")
        ZZ = correlation_matrix(ψ, "Z", "Z")

        XX = cpu(XX)
        YY = cpu(YY)
        ZZ = cpu(ZZ)
        
        # st = [isodd(n) ? "1" : "0" for n = 1:N]
        # ψ = randomMPS(BigFloat, sites, st)
        ψ = cpu(ψ)
        ψ = dense(ψ)
        # LinearAlgebra.normalize!(ψ)


        try
            # @printf("Sampling\n")
            # samples = getsamples(LinearAlgebra.normalize(ψ), randombases(N, nshots), 1) # nshots different meas. bases, each base with one shot
            """for i in ProgressBar(1:nshots)
                ψₛ = ψ
                ψₛ = gpu(ψₛ)
                v1 = sample!(dense(ψₛ))
            end"""

            @bp

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

            fout = h5open(basepath * "simulation_$(Lx)x$(Ly)_id$(k)_gs.jld", "w")
            write(fout, "psi", ψ)
            close(fout)
        catch e
            showerror(stdout, e)
            @printf("Sampling failed for sample %i\n", k)
            # saving error states
            path = basepath * "failed_simulation_$(Lx)x$(Ly)_id$(k).jld"
            JLD.jldopen(path, "w") do fout
                JLD.write(fout, "J", J)
                JLD.write(fout, "ZZ", ZZ)
                JLD.write(fout, "YY", YY)
                JLD.write(fout, "XX", XX)
                JLD.write(fout, "E", E)
                #JLD.write(fout, "SvN", SvN)
            end

            fout = h5open(basepath * "failed_simulation_$(Lx)x$(Ly)_id$(k)_gs.jld", "w")
            write(fout, "psi", ψ)
            close(fout)
        end
        println("")
        # println("samples:")
        # display(samples)
        # println("psi:")
        # display(ψ)
        # println("XX:")
        # display(XX)
        groundstate = ψ

    end
    return groundstate
end

# print(@code_string main())

if "" != PROGRAM_FILE && realpath(@__FILE__) == realpath(PROGRAM_FILE)
    main()
end

