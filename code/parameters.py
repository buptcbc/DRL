import math
import job

class parameters:

    def __init__(self):

        self.output_filename = 'data/tmp'

        self.num_epochs = 10000 # number of training
        self.simu_len = 50      # simulation job numbers
        self.num_ex =1          # simu_len group number

        self.output_freq = 10   # interval: output and store parameter

        self.num_seq_per_batch = 10 #parallel network.
        self.episode_max_length = 200 # enforce an artificial termination

        self.num_user = 5         # user number

        self.time_horizon = 20  # number of time steps in graph
        self.max_job_len = 15   # job computing resource length
        self.res_slot = 10      # can consume network resource
        self.max_req_slot = 10  # max request computing resource number
        self.mec_cap = 20       # shared computing resource

        self.backlog_size = 60      # queue : store work

        self.max_track_since_new = 10    # track how many time steps since new  jobs
        self.job_col_num = 40   # max color number in the graph

        self.new_job_rate = 0.1  # new job generate prob

        self.discount = 1       # discount factor

        self.dist = job.Dist(self.max_job_len,self.max_req_slot,self.res_slot)

        # graph represent
        assert self.backlog_size % self.time_horizon==0 # can represent graph
        self.backlog_width =int( math.ceil(self.backlog_size / self.time_horizon) )
        self.network_input_height = self.time_horizon
        self.network_input_width = self.mec_cap + self.res_slot +\
                                   (self.max_req_slot + self.res_slot + self.res_slot + self.backlog_width )* self.num_user + 1
        # shared mec computing resource, shared network resource,task's computing and network resource ,
        # user local computing resource ,backlog queue.
        # + 1 is extra information

        self.network_output_dim = self.num_user + 1  # 1 is for void action

        self.delay_penalty = -1     # delay penalty
        self.hold_penalty = -1
        self.dismiss_penalty = -1

        self.num_frames = 1         # number of frames to combine and process
        self.lr_rate = 0.001        # learning rate
        self.rms_rho = 0.9          # rms_rho
        self.rms_eps = 1e-9         # rms_eps

        self.unseen = False

        self.batch_size = 10

        self.evaluate_policy_name = 'SJF'








