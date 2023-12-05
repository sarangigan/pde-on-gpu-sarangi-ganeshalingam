using Plots
using Literate

"""
## save_array

Save a binary array with a specific name in a file in your current working directory.

Insert the specified name and a matrix, when calling the function.

### Syntax
save_array(Aname,A)
"""
function save_array(Aname,A)
    fname = string(Aname,".bin")
    out = open(fname,"w"); write(out,A); close(out)
end

"""
## load_array
Load a binary array with a specified name.

### Syntax
load_array(Aname,A)
"""
function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end

function main()

    A = rand(3, 3)

    B = zeros(3, 3)
    
    save_array("random_array", A)

    load_array("random_array", B)

    return B
end

B = main()
heatmap(B)