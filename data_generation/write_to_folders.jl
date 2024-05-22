
using HDF5
using JLD
using ProgressBars
using Debugger
using CodeTracking
using CSV
using DataFrames
using LinearAlgebra
using ArgParse
using ITensors

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


function jld_to_txt()
    base_path = "final_data/unif_nonlocal/"
    data_path = "../data_nonlocal/"
    for Lx=4:6
        println(Lx)
        folder = "data_$(Lx)x5/"
        for id=1:4096
            path_save = base_path * folder * "simulation_$(Lx)x5_id$(id)_"
            fp = data_path * "simulation_$(Lx)x5_id$(id).jld"
            if !isfile(fp)
                continue
            end
            file = JLD.jldopen(fp, "r") 
            XX = reshape(real(read(file, "XX")), (Lx*5, Lx*5))
            YY = reshape(real(read(file, "YY")), (Lx*5, Lx*5))
            ZZ = reshape(real(read(file, "ZZ")), (Lx*5, Lx*5))
            J = read(file, "J")
            J = reshape(J, length(J), 1)
            
            CSV.write(path_save * "XX.txt", DataFrame(XX, :auto), header=false, delim="\t")
            CSV.write(path_save * "YY.txt", DataFrame(YY, :auto), header=false, delim="\t")
            CSV.write(path_save * "ZZ.txt", DataFrame(ZZ, :auto), header=false, delim="\t")
            CSV.write(path_save * "couplings.txt", DataFrame(J, :auto), header=false, delim="\t")
        end
    end
end

function main()
    

    parsed_args = parse_commandline()
    J = CSV.read("lds_sequences/lds_4x5.txt", DataFrame, header=false)
    J = Matrix(J)
    #F = J[1]
    for j in 1:1
        println(J[1])
    end

    println(parsed_args["Lx"])

    Lx = 9
    Ly = 5
    k = 1004
    basepath = "data_generation/data_test/"

    lattice = square_lattice(Lx, Ly; yperiodic=false)
    for (b, bond) in enumerate(lattice)
        println(bond.s1, ", ", bond.s2)
    end

    @bp

    path = basepath * "simulation_$(Lx)x$(Ly)_id$(k).jld"
    file = JLD.jldopen(path, "r") 
    # println(reshape(real(read(file, "ZZ")), (45, 45)))
    A = reshape(real(read(file, "ZZ")), (45, 45))
    close(file)

    f = open("new_data/data_9x5/simulation_9x5_id2_ZZ.txt")
    C = CSV.read("new_data/data_9x5/simulation_9x5_id2_ZZ.txt", DataFrame, header=false, delim="\t")
    B = Matrix(C)

    println(minimum(A-B))
    println(maximum(A-B))
    # println(B[5])

    return Nothing
end

# print(@code_string main())

if "" != PROGRAM_FILE && realpath(@__FILE__) == realpath(PROGRAM_FILE)
    jld_to_txt()
end
