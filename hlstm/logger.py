import os.path
import pandas as pd
import copy


class Logger():

    def __init__(self):
        self.logs = list()
        self.sess = -1

    def start_session(self, sess_name=''):
        if self.sess != -1:
            past_sess = self.sess
            self.stop_session()
        self.sess = len(self.logs)
        if not sess_name:
            sess_name = str(self.sess)
        self.sess_name = sess_name
        self.current_logs = list()
        print("session number: %i with name: %s started" %
              (self.sess, self.sess_name))

    def stop_session(self):
        if self.current_logs:
            self.logs += self.current_logs
            print("session number: %i with name: %s stoped and recorded" %
                  (self.sess, self.sess_name))
        else:
            print("EMPTY session %i with name: %s stoped and NOT recorded"
                  % (self.sess, self.sess_name))
        self.current_logs = list()
        self.sess = -1

    def record_train_logs(self, model_properies, train_properties,
                          train_results, train_set_info=None, dev_set_info=None):
        if self.sess != -1:
            if not train_set_info:
                train_set_info = dict()
            if not dev_set_info:
                dev_set_info = dict()
            logs_dict = copy.deepcopy(model_properies)
            logs_dict.update(copy.deepcopy(train_properties))
            logs_dict.update(copy.deepcopy(train_results))
            logs_dict.update({('TRAIN_' + key): value for key,
                              value in copy.deepcopy(train_set_info).items()})
            logs_dict.update({('DEV_' + key): value for key,
                              value in copy.deepcopy(dev_set_info).items()})
            logs_dict["SESS_NAME"] = self.sess_name
            logs_dict["SESS_NUM"] = self.sess  # like hash
            self.current_logs.append(logs_dict)
            return self.current_logs
        else:
            print("Session does not exist")
            return -1

    def save_logs(self, path, append=True):
        df = self.get_all_logs_dataframe()
        path = path+'.dfpickle'
        if append == True and os.path.isfile(path):
            df_old = pd.read_pickle(path)
            df = pd.concat([df_old, df], ignore_index=True)
            df = df.iloc[df.astype(str).drop_duplicates().index]
        df.to_pickle(path)
        if append == True:
            print('Logs appended to file: %s' % path)
        else:
            print('Logs saved to file: %s' % path)

    def get_all_logs_dataframe(self):
        return pd.DataFrame(self.get_all_logs())

    def get_all_logs(self):
        if self.sess != -1:
            return self.logs + self.current_logs
        else:
            return self.logs
