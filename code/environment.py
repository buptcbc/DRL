import numpy as np
import math
import matplotlib.pyplot as plt

import parameters

class environment:
    def __init__(self,pa,nw_mec_len_seqs=None,nw_net_len_seqs=None,nw_size_seqs=None,seed=42,render=False,end='no_new_job'):
        self.pa = pa
        self.render =render
        self.end = end  # termination type. All_done or No_new_job

        self.nw_dist = pa.dist.bi_dist

        self.cur_time = 0

        # set up random seed
        if self.pa.unseen:
            np.random.seed(314159)
        else:
            np.random.seed(seed)

        # user work generate.
        if nw_net_len_seqs is None or nw_size_seqs is None or nw_mec_len_seqs is None:
            # generate work
            self.nw_mec_len_seqs ,self.nw_net_len_seqs, self.nw_size_seqs = self.generate_work()
        else:
            self.nw_net_len_seqs = nw_net_len_seqs # it is matrix .  row is user_id, col is index.
            self.nw_mec_len_seqs = nw_mec_len_seqs # it is matrix .  row is user_id, col is index.
            self.nw_size_seqs = nw_size_seqs       # it is matrix .  row is user_id, col is index. mec cap.

        # self.seq_user = np.zeros(self.pa.num_user,dtype=int)   # which user sequence
        self.seq_inx = 0  # index in that user
        self.seq_ex = 0   # index in ex_num

        self.job_record = JobRecord()       # record job schedule information
        self.machine = Machine(pa)          # shared net and mec resource information
        self.job_slot = JobSlot(pa)         # user job
        self.extra_info = Extra_info(pa)    # no job arrive.time information

        self.local_cpus = []      # cpu information in community
        self.job_backlogs = []    # job_backlog information  in  community

        for i in range(self.pa.num_user):
            self.local_cpu = LocalCpu(pa)       # local cpu resorce information
            self.job_backlog = Job_backlog(pa)  # user job cache queue
            self.local_cpus.append(self.local_cpu)
            self.job_backlogs.append(self.job_backlog)

    def generate_work(self):
        # this is used to generate user work.
        # it is matrix.  row is user_id, col is index.
        simu_len = self.pa.num_ex * self.pa.num_user * self.pa.simu_len
        nw_mec_len_seqs = np.zeros(simu_len, dtype=int)
        nw_net_len_seqs = np.zeros(simu_len,dtype=int)
        nw_seq_size = np.zeros(simu_len, dtype=int)

        for i in range(simu_len):
            if np.random.rand() < self.pa.new_job_rate:  # a new job generate
                nw_mec_len_seqs[i], nw_net_len_seqs[i], nw_seq_size[i] = self.nw_dist()

        nw_mec_len_seqs = np.reshape(nw_mec_len_seqs, [self.pa.num_ex,self.pa.num_user, self.pa.simu_len])
        nw_net_len_seqs = np.reshape(nw_net_len_seqs, [self.pa.num_ex,self.pa.num_user,self.pa.simu_len])
        nw_seq_size =     np.reshape(nw_seq_size, (self.pa.num_ex,self.pa.num_user, self.pa.simu_len))

        return nw_mec_len_seqs, nw_net_len_seqs, nw_seq_size

    def get_new_job_from_seq(self,num_ex,nw_user,seq_idx):
        new_job = Job(res=self.nw_size_seqs[num_ex,nw_user,seq_idx],
                      job_mec_len = self.nw_mec_len_seqs[num_ex,nw_user,seq_idx],
                      job_net_len = self.nw_net_len_seqs[num_ex,nw_user,seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time= self.cur_time)
        return new_job
    def observe(self):

        backlog_width = int(math.ceil(self.pa.backlog_size / float(self.pa.time_horizon)))

        img_re = np.zeros(( int(self.pa.network_input_height), int(self.pa.network_input_width) ),dtype= int)

        img_ptr = 0

        # machine represent
        img_re[:,img_ptr:img_ptr+self.pa.mec_cap] = self.machine.canvas_mec    # mec represent
        img_ptr = img_ptr + self.pa.mec_cap
        img_re[:, img_ptr:img_ptr+self.pa.res_slot] = self.machine.canvas_net  # net represent
        img_ptr = img_ptr + self.pa.res_slot

        # user represent
        for i in range(self.pa.num_user):
            # user job represent
            if self.job_slot.slot[i] is not None:
                img_re[:self.job_slot.slot[i].job_mec_len, img_ptr:img_ptr + self.job_slot.slot[i].res] = 1 # job mec represent
                img_ptr = img_ptr + self.pa.res_slot
                img_re[:self.job_slot.slot[i].job_net_len, img_ptr:img_ptr + self.pa.res_slot ] = 1                    # job net represent
                img_ptr = img_ptr +self.pa.res_slot
            else:
                img_ptr = img_ptr + self.pa.res_slot * 2
            # user local cpu represent

            img_re[:, img_ptr:img_ptr + self.pa.res_slot] = self.local_cpus[i].canvas_local
            img_ptr = img_ptr + self.pa.res_slot

            # user cache job queue
            img_re[: int(self.job_backlogs[i].cur_size / backlog_width) ,
            img_ptr:img_ptr + backlog_width ] = 1

            if self.job_backlogs[i].cur_size % backlog_width > 0:
                img_re[int(self.job_backlogs[i].cur_size / backlog_width) ,
                img_ptr: img_ptr + self.job_backlogs[i].cur_size % backlog_width] = 1

            img_ptr = img_ptr + backlog_width

        # extra information

        img_re[:,img_ptr:img_ptr + 1 ] = self.extra_info.time_since_last_new_job / \
            float(self.extra_info.max_track_time_since_last_job)
        img_ptr += 1
        # print(img_ptr,img_re.shape[1])
        assert img_ptr == img_re.shape[1]

        return img_re

    def plot_state(self):
        plt.figure("screen",figsize=(20,5))

        ptr = 1
        # computing resource
        plt.subplot(4,self.pa.num_user + 1,ptr)
        plt.imshow(self.machine.canvas_mec,interpolation='nearest',vmax=1)
        for i in range(self.pa.num_user):

            job_slot = np.zeros((self.pa.time_horizon,self.pa.res_slot),dtype = int)
            if self.job_slot.slot[i] is not None:
                job_slot[:self.job_slot.slot[i].job_mec_len, :self.job_slot.slot[i].res] = 1
            ptr +=1
            plt.subplot(4, self.pa.num_user + 1, ptr)
            plt.imshow(job_slot,interpolation='nearest',vmax=1)

        # net resource
        ptr += 1
        plt.subplot(4,self.pa.num_user + 1, self.pa.num_user + 2)
        plt.imshow(self.machine.canvas_net,interpolation='nearest',vmax=1)
        for i in range(self.pa.num_user):
            job_slot = np.zeros((self.pa.time_horizon,self.pa.res_slot),dtype = int)
            if self.job_slot.slot[i] is not None:
                job_slot[:self.job_slot.slot[i].job_net_len,:] = 1
            ptr += 1
            plt.subplot(4,self.pa.num_user + 1,ptr)
            plt.imshow(job_slot,interpolation='nearest',vmax=1)

        #local cpu
        ptr += 1
        for i in range(self.pa.num_user):
            ptr += 1
            plt.subplot(4,self.pa.num_user + 1,ptr)
            plt.imshow(self.local_cpus[i].canvas_local , interpolation='nearest',vmax=1)

        # Extro info
        ptr += 1
        extro_info = np.ones((self.pa.time_horizon,1),dtype =int)*\
            self.extra_info.time_since_last_new_job / \
            float(self.extra_info.max_track_time_since_last_job)
        plt.subplot(4,self.pa.num_user + 1,ptr)
        plt.imshow(extro_info ,interpolation='nearest')

        #user backlog
        backlog_width = int(math.ceil(self.pa.backlog_size / self.pa.time_horizon))
        for i in range(self.pa.num_user):
            ptr += 1
            backlog = np.zeros((self.pa.time_horizon,backlog_width))

            backlog[:int(self.job_backlogs[i].cur_size / backlog_width), :backlog_width] = 1
            backlog[int(self.job_backlogs[i].cur_size / backlog_width),\
                :self.job_backlogs[i].cur_size % backlog_width] = 1
            plt.subplot(4,self.pa.num_user + 1,ptr)
            plt.imshow(backlog,interpolation='nearest',vmax = 1)

        plt.show()



    def get_reward(self):

        reward = 0

        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.job_mec_len)

        for i in range(self.pa.num_user):
            for j in self.local_cpus[i].running_job :
                reward += self.pa.delay_penalty / float(j.job_mec_len)

        for j in self.job_slot.slot:
            if j is not None :
                reward += self.pa.hold_penalty / float(j.job_mec_len)

        for i in range(self.pa.num_user):
            for j in self.job_backlogs[i].backlog :
                if j is not None:
                    reward += self.pa.dismiss_penalty /  float(j.job_mec_len)

        return reward

    def step(self,a):

        status = None
        done = False
        reward = 0
        info = None

        if a == self.pa.num_user : # void action
            status = 'Move On'
        elif self.job_slot.slot[a] is None: # job is void. So this is a void action
            status = 'Move On'
        else :
            allocated= self.machine.allocate_job(self.job_slot.slot[a],self.cur_time)

            if not allocated : # job can't schedule mec .
                # test job schedule local cpu.
                Allocated = self.local_cpus[a].allocate_job(self.job_slot.slot[a],self.cur_time)
                #schedule local cpu
                # print('local',Allocated)
                if  Allocated :
                    status = 'Allocate '
                    # print('local')
                # can't schedule local cpu
                else:
                    status = 'Move On'
            else :
                status = 'Allocate'

        # fisrt status==move on,
        # then update time information,
        # job sequence number information
        # test done and termination
        # schedule job and job come to job_slot or backlog
        # update extro_information

        if status == 'Move On':  # move on a time stamp
            self.cur_time += 1
            # local cpu move on
            for i in range(self.pa.num_user):
                self.local_cpus[i].proceed_time(self.cur_time)
            # machine move on
            self.machine.time_proceed(self.cur_time)
            # Extra move on
            self.extra_info.time_proceed()

            # new job arrive
            self.seq_inx += 1

            if self.end == 'no_new_job' : # end of no new job
                if self.seq_inx >= self.pa.simu_len :
                    done = True
            elif self.end == 'All_done' :

                if self.seq_inx >= self.pa.simu_len and \
                    len(self.machine.running_job) ==0 and \
                    all(s is None for s in self.job_slot.slot) and \
                    all(len(s.running_job)==0 for s in self.local_cpus) and \
                    all(s.backlog is None for s in self.job_backlogs):

                    done = True
                elif self.cur_time > self.pa.episode_max_length :  # run too long.Force termination
                    done = True

            if not done :

                if self.seq_inx < self.pa.simu_len : # otherwise :no new job

                    # user in community
                    for i in range(self.pa.num_user) :
                        new_job = self.get_new_job_from_seq(self.seq_ex,i,self.seq_inx)

                        if new_job.job_mec_len > 0: # new job come

                            to_backlog = True
                            if self.job_slot.slot[i] is None :
                                self.job_slot.slot[i] = new_job
                                self.job_record.record[new_job.id] = new_job # save dictionary
                                to_backlog = False

                            if to_backlog :
                                if self.job_backlogs[i].cur_size < self.pa.backlog_size:
                                    self.job_backlogs[i].backlog[self.job_backlogs[i].cur_size] = new_job
                                    self.job_backlogs[i].cur_size += 1
                                    self.job_record.record[new_job.id] = new_job  # save dictionary
                                else:
                                    print('user',i,'backlog_full')

                            self.extra_info.new_job_comes()

            reward = self.get_reward()

        elif status == 'Allocate':
            self.job_record.record[self.job_slot.slot[a].id] = self.job_slot.slot[a] #allocate job
            self.job_slot.slot[a] = None

            # dequeue backlog
            if self.job_backlogs[a].cur_size > 0 :
                self.job_slot.slot[a] = self.job_backlogs[a].backlog[0]
                self.job_backlogs[a].backlog[:-1] = self.job_backlogs[a].backlog[1:]
                self.job_backlogs[a].backlog[-1] = None
                self.job_backlogs[a].cur_size -= 1

        ob = self.observe()

        info = self.job_record

        if done :

            self.seq_inx = 0
            self.seq_ex = self.seq_ex + 1
            self.reset()

        if self.render:
            self.plot_state()
        #print(done,status)
        return ob,reward,done,info

    def reset(self):
        self.seq_inx = 0
        self.cur_time = 0

        # initial system
        self.job_record = JobRecord()  # record job schedule information
        self.machine = Machine(self.pa)  # shared net and mec resource information
        self.job_slot = JobSlot(self.pa)  # user job
        self.extra_info = Extra_info(self.pa)  # no job arrive.time information

        self.local_cpus = []  # cpu information in community
        self.job_backlogs = []  # job_backlog information  in  community

        for i in range(self.pa.num_user):
            self.local_cpu = LocalCpu(self.pa)  # local cpu resorce information
            self.job_backlog = Job_backlog(self.pa)  # user job cache queue
            self.local_cpus.append(self.local_cpu)
            self.job_backlogs.append(self.job_backlog)


class Job:
    def __init__(self,res,job_mec_len,job_net_len,job_id,enter_time):
        self.id = job_id
        self.res = res
        self.job_mec_len = job_mec_len
        self.job_net_len = job_net_len
        self.enter_time =enter_time
        self.start_time = -1
        self.end_time = -1

class JobRecord:
    def __init__(self):
        self.record = {}

class Machine:
    def __init__(self,pa): #machine is used to allocate job in shared resource
        self.num_res = 2
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.mec_cap = pa.mec_cap

        # this is avaliable resource slot
        self.avl_mec_slot = np.ones(self.time_horizon) * self.mec_cap
        self.avl_net_slot = np.ones(self.time_horizon) * self.res_slot

        self.running_job = [] # it represent running job

        # color that represent job
        self.colormap = np.arange( 1 / float(pa.job_col_num),1 , 1 /float(pa.job_col_num))
        np.random.shuffle(self.colormap)
        # graph represent
        self.canvas_mec = np.zeros((self.time_horizon,self.mec_cap))  # mec computing resource
        self.canvas_net = np.zeros((self.time_horizon,self.res_slot)) # net computing resource

    def allocate_job(self,job,curr_time):

        allocated = False

        # allocate net resource , then allocate mec compute resource
        for i in range(0,self.time_horizon-job.job_net_len):
            #allocate net resource
            new_avl_net_slot =self.avl_net_slot[i:i+job.job_net_len] - self.res_slot
            if np.all(new_avl_net_slot[:] >= 0):   # net source allocate successfully

                used_color = np.unique(self.canvas_net[:])
                for color in self.colormap:
                    if color not in used_color:
                        new_color = color
                        break

                for j in range(i+job.job_net_len,self.time_horizon-job.job_mec_len):
                    new_avl_mec_slot = self.avl_mec_slot[j:j+job.job_mec_len] - job.res
                    if np.all(new_avl_mec_slot[:] >= 0):    # mec source allocate successfully

                        allocated = True

                        self.avl_net_slot[i:i+job.job_net_len] = new_avl_net_slot
                        self.avl_mec_slot[j:j+job.job_mec_len] = new_avl_mec_slot

                        job.start_time = curr_time + i
                        job.end_time = curr_time + j + job.job_mec_len

                        self.running_job.append(job)

                        # update graph represent



                        assert job.start_time != -1
                        assert job.end_time != -1
                        assert job.end_time > job.start_time

                        canvas_net_start = i
                        canvas_net_end = i + job.job_net_len
                        canvas_mec_start =j
                        canvas_mec_end =j + job.job_mec_len

                        for t in range(canvas_net_start,canvas_net_end):
                            self.canvas_net[t,:] = new_color   # net graph

                        for t in range(canvas_mec_start,canvas_mec_end):
                            avl_slot = np.where(self.canvas_mec[t,:] == 0)[0]
                            self.canvas_mec[t,avl_slot[:job.res]] = new_color # mec graph

                        break

                if allocated == True :
                    break
        return allocated

    def time_proceed(self,curr_time):

        self.avl_mec_slot[:-1] = self.avl_mec_slot[1:]
        self.avl_net_slot[:-1] = self.avl_net_slot[1:]
        self.avl_mec_slot[-1] = self.mec_cap
        self.avl_net_slot[-1] = self.res_slot

        for job in self.running_job :
            if job.end_time <= curr_time:
                self.running_job.remove(job)

        # update mec and net graph

        self.canvas_net[:-1,:] = self.canvas_net[1:,:]
        self.canvas_mec[:-1,:] = self.canvas_mec[1:,:]
        self.canvas_net[-1,:] = 0
        self.canvas_mec[-1,:] = 0

class JobSlot:
    # this is used to represent user's job .it contain job class.
    def __init__(self,pa):
        self.slot = [None] * pa.num_user

class LocalCpu:
    # this is used to represent local computing.
    def __init__(self,pa):
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot
        self.avl_slot = np.ones(self.time_horizon,dtype=int) * self.res_slot

        self.running_job = [] # it represent running job in local

        self.canvas_local = np.zeros((self.time_horizon,self.res_slot),dtype=int)
    def allocate_job(self,job,cur_time):
        allocated = False

        for t in range(0,self.time_horizon - job.job_mec_len):

            new_avl_slot = self.avl_slot[t:t+job.job_mec_len] - job.res

            if np.all(new_avl_slot[:]>=0) :

                allocated = True
                # update avl_slot
                self.avl_slot[t:t+job.job_mec_len] = new_avl_slot
                job.start_time = cur_time + t
                job.end_time = job.start_time + job.job_mec_len

                self.running_job.append(job)
                # update local cpu graph
                for i in range(t,t+job.job_mec_len):
                    avl_slot = np.where(self.canvas_local[i,:]==0)[0]
                    self.canvas_local[i,avl_slot[:job.res]] = 1

                break

        return allocated

    def proceed_time(self,cur_time):

        self.avl_slot[:-1] = self.avl_slot[1:]
        self.avl_slot[-1] = self.res_slot

        for job in self.running_job:
            if job.end_time <= cur_time:
                self.running_job.remove(job)

        # update graph

        self.canvas_local[:-1,:] = self.canvas_local[1:,:]
        self.canvas_local[-1,:] = 0

class Job_backlog:

    def __init__(self,pa):
        self.backlog = [None] * pa.backlog_size
        self.cur_size = 0

class Extra_info:
    def __init__(self,pa):
        self.time_since_last_new_job = 0
        self.max_track_time_since_last_job = pa.max_track_since_new

    def new_job_comes(self):
        self.time_since_last_new_job = 0

    def time_proceed(self):
        if self.time_since_last_new_job < self.max_track_time_since_last_job :
            self.time_since_last_new_job += 1


def test_backlog():
    pa = parameters.parameters()
    pa.num_nw = 5
    pa.simu_len = 20  # 50
    pa.new_job_rate = 1
    env = environment(pa,render= True)
    for i in range(15):
        print(i)
        env.step(5)

    for i in range(15):
        print(i)
        env.step(4)


    print("New job is backlogged.")
if __name__ == "__main__":
    test_backlog()
