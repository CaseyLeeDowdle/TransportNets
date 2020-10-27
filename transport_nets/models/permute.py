def channel(output_dim):
    return list(range(0,output_dim,2)) + list(range(1,output_dim,2))

def flip(output_dim):
    return list(range(output_dim))[::-1]