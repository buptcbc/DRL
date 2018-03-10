import numpy as np
import math
import parameters
class Dist:

    def __init__(self,num_nw_max_len,num_nw_max_size,num_net_max_size):
        self.num_nw_max_len = num_nw_max_len
        self.num_nw_max_size = num_nw_max_size
        self.num_net_max_size = num_net_max_size

        self.job_small_chance = 0.8

        self.job_len_big_upper = num_nw_max_len
        self.job_len_big_lower = num_nw_max_len * 2 / 3
        self.job_len_small_upper = num_nw_max_len / 5
        self.job_len_small_lower = 1

        self.job_size_upper = num_nw_max_size
        self.job_size_lower = num_nw_max_size / 2


    def bi_dist(self):

        #--job length that consume computing resource --
        if np.random.rand() < self.job_small_chance :
            nw_len = np.random.randint(self.job_len_small_lower,self.job_len_small_upper)
        else: # big job length
            nw_len = np.random.randint(self.job_len_big_lower,self.job_len_big_upper)

        nw_size = np.random.randint(self.job_size_lower,self.job_size_upper+1)

        #-- job length that consume network resource --

        cap = nw_len*nw_size /5 # 5 is used to scale factor.
        net_len = math.ceil(cap / self.num_net_max_size )

        return nw_len, net_len, nw_size

def generate_work_sequence(pa,seed=42):
    # this use new_job rate large np.random to generate new job.
    # new job generate small job or large job by bi_model
    np.random.seed(seed)
    nw_dist = pa.dist.bi_dist
    simu_len =  pa.num_ex * pa.num_user * pa.simu_len
    nw_mec_len_seqs = np.zeros(simu_len,dtype=int)
    nw_net_len_seqs = np.zeros(simu_len,dtype=int)
    nw_seq_size = np.zeros(simu_len,dtype=int)

    for i in range(simu_len):
        if np.random.rand() < pa.new_job_rate : # a new job generate
            nw_mec_len_seqs[i],nw_net_len_seqs[i],nw_seq_size[i] = nw_dist()

    nw_mec_len_seqs = np.reshape(nw_mec_len_seqs,[pa.num_ex,pa.num_user,pa.simu_len])
    nw_net_len_seqs = np.reshape(nw_net_len_seqs,[pa.num_ex,pa.num_user,pa.simu_len])
    nw_seq_size = np.reshape(nw_seq_size,(pa.num_ex,pa.num_user,pa.simu_len))

    return nw_mec_len_seqs,nw_net_len_seqs, nw_seq_size
if __name__ == "__main__":
    pa = parameters.parameters()
    pa.num_user = 5
    pa.simu_len = 7
    pa.new_job_rate = 0.2
    nw_mec_len_seqs, nw_net_len_seqs, nw_seq_size = generate_work_sequence(pa)
    print(nw_mec_len_seqs)
    print(nw_net_len_seqs)
    print(nw_seq_size)