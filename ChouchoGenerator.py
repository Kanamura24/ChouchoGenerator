#!/usr/bin/env python
# coding: utf-8


"""
 @file ChouchoGenerator.py
 @brief Choucho Musubi Motion Generator using pytorch RT Component
 @date $Date$


"""
import sys
import time

sys.path.append(".")

# Import RTM module
import RTC
import OpenRTM_aist

import ManipulatorCommonInterface_DataTypes_idl
import ManipulatorCommonInterface_Common_idl
import ManipulatorCommonInterface_MiddleLevel_idl
import Img

from IPython.core.debugger import Tracer; keyboard = Tracer()
from PIL import Image
import numpy as np
import cv2


import os, math, re
import datetime


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import cPickle as pickle


from chou_models.lstm import LSTM
from chou_models.cae import CAE as ae
from chou_models.trainer import dataClass
# import matplotlib.pyplot as plt

# Import Service implementation class
# <rtc-template block="service_impl">

# </rtc-template>

# Import Service stub modules
# <rtc-template block="consumer_import">
import JARA_ARM, JARA_ARM__POA
import JARA_ARM_LEFT, JARA_ARM_LEFT__POA


# </rtc-template>
label_dcnn = False

lstm_series_idx = 0
img_dim = 30 #40 ->128

total_step = 30000
#Dur = 0.3
Dur = 2.0
saveflag=True
calibrate_gripper=False

# This module's spesification
# <rtc-template block="module_spec">
armimagepredictor_keras_spec = ["implementation_id", "ChouchoGenerator",
         "type_name",         "ChouchoGenerator",
         "description",       "Choucho Musubi Motion Generator using pytorch RT Component",
         "version",           "1.0.0",
         "vendor",            "kanamura",
         "category",          "Experimental",
         "activity_type",     "STATIC",
         "max_instance",      "1",
         "language",          "Python",
         "lang_type",         "SCRIPT",
         "conf.default.debug", "1",
         "conf.default.gripper_close_ratio", "0.1",

         "conf.__widget__.debug", "text",

         "conf.__type__.debug", "int",

         "conf.__widget__.gripper_close_ratio", "text",

         "conf.__type__.gripper_close_ratio", "int",

         ""]
# </rtc-template>

##
# @class ChouchoGenerator
# @brief Arm Image Predictor using Keras RT Component
#
#
class ChouchoGenerator(OpenRTM_aist.DataFlowComponentBase):

    ##
    # @brief constructor
    # @param manager Maneger Object
    #
    def __init__(self, manager):
        OpenRTM_aist.DataFlowComponentBase.__init__(self, manager)

        camera_arg = [None] * ((len(Img._d_TimedCameraImage) - 4) / 2)
        self._d_camera = Img.TimedCameraImage(*camera_arg)
        """
        """
        self._cameraIn = OpenRTM_aist.InPort("camera", self._d_camera)

        """
        """
        self._manipCommon_RPort = OpenRTM_aist.CorbaPort("manipCommon_R")
        """
        """
        self._manipMiddle_RPort = OpenRTM_aist.CorbaPort("manipMiddle_R")



        self._manipCommon_LPort = OpenRTM_aist.CorbaPort("manipCommon_L")
        """
        """
        self._manipMiddle_LPort = OpenRTM_aist.CorbaPort("manipMiddle_L")

        """
        """
        self._manipCommon_R = OpenRTM_aist.CorbaConsumer(interfaceType=JARA_ARM.ManipulatorCommonInterface_Common)
        """
        """
        self._manipMiddle_R = OpenRTM_aist.CorbaConsumer(interfaceType=JARA_ARM.ManipulatorCommonInterface_Middle)
        """
        """
        self._manipCommon_L = OpenRTM_aist.CorbaConsumer(interfaceType=JARA_ARM_LEFT.ManipulatorCommonInterface_Common)
        """
        """
        self._manipMiddle_L = OpenRTM_aist.CorbaConsumer(interfaceType=JARA_ARM_LEFT.ManipulatorCommonInterface_Middle)

        # initialize of configuration-data.
        # <rtc-template block="init_conf_param">
        """

         - Name:  debug
         - DefaultValue: 1
        """
        self._debug = [1]

        self._model = None

        """

         - Name:  gripper_close_ratio
         - DefaultValue: 0.1
        """
        self._gripper_close_ratio = [0.1]

        self._model = None

        # </rtc-template>



    ##
    #
    # The initialize action (on CREATED->ALIVE transition)
    # formaer rtc_init_entry()
    #
    # @return RTC::ReturnCode_t
    #
    #
    def onInitialize(self):
        # Bind variables and configuration variable
        self.bindParameter("debug", self._debug, "1")
        self.bindParameter("gripper_close_ratio", self._gripper_close_ratio, "0.1")

        # Set InPort buffers
        self.addInPort("camera",self._cameraIn)

        # Set OutPort buffers

        # Set service provider to Ports

        # Set service consumers to Ports
        self._manipCommon_RPort.registerConsumer("JARA_ARM_ManipulatorCommonInterface_Common", "JARA_ARM::ManipulatorCommonInterface_Common", self._manipCommon_R)
        self._manipMiddle_RPort.registerConsumer("JARA_ARM_ManipulatorCommonInterface_Middle", "JARA_ARM::ManipulatorCommonInterface_Middle", self._manipMiddle_R)

        self._manipCommon_LPort.registerConsumer("JARA_ARM_LEFT_ManipulatorCommonInterface_Common", "JARA_ARM_LEFT::ManipulatorCommonInterface_Common", self._manipCommon_L)
        self._manipMiddle_LPort.registerConsumer("JARA_ARM_LEFT_ManipulatorCommonInterface_Middle", "JARA_ARM_LEFT::ManipulatorCommonInterface_Middle", self._manipMiddle_L)

        # Set CORBA Service Ports
        self.addPort(self._manipCommon_RPort)
        self.addPort(self._manipMiddle_RPort)
        self.addPort(self._manipCommon_LPort)
        self.addPort(self._manipMiddle_LPort)
        return RTC.RTC_OK

    #   ##
    #   #
    #   # The finalize action (on ALIVE->END transition)
    #   # formaer rtc_exiting_entry()
    #   #
    #   # @return RTC::ReturnCode_t
    #
    #   #
    #def onFinalize(self):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The startup action when ExecutionContext startup
    #   # former rtc_starting_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onStartup(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The shutdown action when ExecutionContext stop
    #   # former rtc_stopping_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onShutdown(self, ec_id):
    #
    #   return RTC.RTC_OK

        ##
        #
        # The activated action (Active state entry action)
        # former rtc_active_entry()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #
    def onActivated(self, ec_id):
        #c = pd.read_csv(os.path.join(dir, 'joints.csv'))
        #Y = [y for y in zip((c['x']-0.12)/0.12, (c['y']+0.12)/0.24, (c['theta']+math.pi)/(2*math.pi))]
        #X = [img_to_array(load_img(os.path.join(dir, png.strip()), target_size=(64,64)))/256 for png in c['ImageFilename']]
        print ('onActivated')
        lstm_model_dir = "./chou_model/"
        lstm_model_name = 'lstm_05000.tar'

        dcnn_model_dir = "./chou_model/"
        dcnn_front_model_name = "front_cae_00300.tar"

        if not os.path.isfile(lstm_model_dir+lstm_model_name):
            print "LSTM file does not exit"
            sys.exit()

        # LSTM model input
        resume_lstm = "./chou_resume/lstm_nn_params.pickle"
        with open(resume_lstm, "rb") as f:
            nn_params_lstm = pickle.load(f)
        #nn_params_lstm["resume"] = resume_lstm
        nn_params_lstm["resume"] = "./chou_model/lstm_05000.tar"
        nn_params_lstm["input_param_test"] = {"mot":0.0, "img":1.0}
        print(nn_params_lstm)

        """
        model_LSTM = LSTM(nn_params_lstm["in_size"], nn_params_lstm["out_size"],\
             nn_params_lstm["c_size"], nn_params_lstm["tau"], nn_params_lstm["variance"])
        """
        #model_LSTM = LSTM(insize, insize, nn_params_lstm["c_size"])
        model_LSTM = LSTM(45, 45, nn_params_lstm["c_size"])
        checkpoint = torch.load(nn_params_lstm["resume"])
        model_LSTM.load_state_dict(checkpoint['model_state_dict'])

        """
        checkpoint = torch.load(nn_params_lstm["resume"])
        print(checkpoint)
        model_LSTM.load_state_dict(checkpoint['model_state_dict'])
        """

        """
        for _ in xrange(1):
            model_LSTM.add_init_c_inter()
        serializers.load_npz(lstm_model_dir + lstm_model_name, model_LSTM)
        model_.set_c_state(range(1), 1, tflag=True)
        """

        # CAE model input (front/left/right)
        ## front
        resume_front_ae = "./chou_resume/cae_nn_params.pickle"
        with open(resume_front_ae, "rb") as f:
            nn_params_front_ae = pickle.load(f)
        print(nn_params_front_ae)
        #nn_params_front_ae["resume"] = resume_front_ae
        nn_params_front_ae["resume"] = "./chou_model/front_cae_00300.tar"
        #nn_params_front_ae["batch"] = 12
        #nn_params_front_ae["gpu"] = 0

        # something
        init_stop = 0
        mot_n=[0]*45
        log_num=0

        # make log file


        # Load min_max_val/val_range
        normalized_range_name = "./chou_resume/min_max_vals.npy"
        with open(normalized_range_name, "rb") as f:
            normalized_range = np.load(f)


        after_range = [-1.0, 1.0]


        # majinai


        print ('nn, OK')


        #self._manipCommon_R._ptr().servoON()
        self._manipMiddle_R._ptr().setSpeedJoint(30)
        #self._manipCommon_L._ptr().servoON()
        self._manipMiddle_L._ptr().setSpeedJoint(30)

        self._manipMiddle_R._ptr().movePTPJointAbs([-math.pi/2.27,-math.pi/1.42, -math.pi/1.33, math.pi/61.86, -math.pi/0.79, math.pi/1.76])
        self._manipMiddle_L._ptr().movePTPJointAbs([math.pi/2.07,-math.pi/3.00, math.pi/1.35, -math.pi/1.12, -math.pi/1.29, -math.pi/1.98])

        self._manipMiddle_R._ptr().moveGripper(100)
        self._manipMiddle_L._ptr().moveGripper(100)

        print ('ur5e, OK')

        return RTC.RTC_OK

        ##
        #
        # The deactivated action (Active state exit action)
        # former rtc_active_exit()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #
    def onDeactivated(self, ec_id):

        cv2.destroyAllWindows()

        return RTC.RTC_OK

        ##
        #
        # The execution action that is invoked periodically
        # former rtc_active_do()
        #
        # @param ec_id target ExecutionContext Id
        #
        # @return RTC::ReturnCode_t
        #
        #

    def normalization(data, indataRange, outdataRange):
        if indataRange[0]!=indataRange[1]:
            data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
            data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
        else:
            data = ( outdataRange[0] +  outdataRange[1]) / 2.
        return data

    def denormalization(data, indataRange, outdataRange):
        if indataRange[0]!=indataRange[1]:
            data = (data - outdataRange[0]) / (outdataRange[1] - outdataRange[0])
            data = data * (indataRange[1] - indataRange[0]) + indataRange[0]
        else:
            data = ( outdataRange[0] +  outdataRange[1]) / 2.
        return data


    def extract(img, params, ae):



        #print("extract IN")
        #print(img)
        #sys.exit()
        test = dataClass(img, size=params["size"], dsize=params["dsize"],
                         batchsize=params["batch"], distort=False, test=True)

        #print(params)
        #sys.exit()

        #model = ae().cuda()
        model = ae()
        #print(params["resume"], params["size"], params["dsize"])
        checkpoint = torch.load(params["resume"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        #test.minibatch_reset(rand=False)
        #test.minibatch_next()
        x_in = test()
        #print(x_in)
        #print(torch.tensor(np.asarray(x_in)).shape)
        #x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in))).cuda()

        #noww = rospy.Time.now()
        #save_img(x_in, "./inp_image/" + 'front-{}-{:0=9}.png'.format(noww.secs, noww.nsecs))

        x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in)))
        f = model.encode(x_in)
        y = model.decode(f)
        #f = f.cpu().detach().numpy().copy()
        #y = y.cpu().detach().numpy().copy()
        f = f.detach().numpy().copy()
        y = y.detach().numpy().copy()

        #print(y)

        #sys.exit()

        #noww = rospy.Time.now()
        #save_img(y, "./rec_image/" + 'front-{}-{:0=9}.png'.format(noww.secs, noww.nsecs))

        return f

    def save_img(img, path):
        img = img[0]
        img *= 255.5
        print(img.shape)
        print(img)
        img = img.transpose(1,2,0)
        cv2.imwrite(path, np.uint8(img))



        #print(y)

        return f
    # ----------------------------------------------------------------------------------------------------------------------- #





    def onExecute(self, ec_id):

        if self._cameraIn.isNew():
            data = self._cameraIn.read()
            w = data.data.image.width
            h = data.data.image.height
            print w, h
            img = np.ndarray(shape=(h, w, 3), dtype=float)
            size = len(data.data.image.raw_data)
            for i in range(w*h):
                img[i/w, i%w, 0] = ord(data.data.image.raw_data[i*3+0])
                img[i/w, i%w, 1] = ord(data.data.image.raw_data[i*3+1])
                img[i/w, i%w, 2] = ord(data.data.image.raw_data[i*3+2])
                pass

            print("Image IN")

            # get raw images
            now = datetime.datetime.now()
            log_path = "/raw_image/" + 'test_{0:%Y%m%d%H%M%S}.png'.format(now)
            #cv2.imwrite(log_path, img)
            #img_front_test = cv2.imread(log_path)
            img_front_test = cv2.resize(img, (480, 300))

            cv2.imwrite(log_path, img_front_test)

            comp_img = extract(img_front_test, nn_params_front_ae, ae)


            result = []

            jpos_L = []
            mot_left = self._manipCommon_L._ptr().getFeedbackPosJoint(jpos_L)  ##TODO get current joint
            print(jpos_L)

            jpos_R = []
            mot_right = self._manipCommon_R._ptr().getFeedbackPosJoint(jpos_R)
            print(jpos_R)

            l_grip = 0.0##TODO gripper isOpen or Close.
            print("l_grip: "+ l_grip)

            r_grip = 0.0##TODO gripper isOpen or Close.
            print("r_grip: "+ r_grip)

            """
            if l_grip > 60:
                l_grip = 0
            elif l_grip <= 60:
                l_grip = 1
            """

            #mot_left = np.hstack((np.asarray(mot_left), np.asarray(mot_left_trq)))
            mot_left = np.asarray(mot_left)
            mot_left = np.append(mot_left, l_grip)
            #mot_right = np.hstack((np.asarray(mot_right), np.asarray(mot_right_trq)))
            mot_right = np.asarray(mot_right)
            mot_right = np.append(mot_right, r_grip)
            mot = np.hstack((np.asarray(mot_right), np.asarray(mot_left)))  ##TODO order is right???

            mot = np.asarray(mot).astype('float32')
            print(mot)
            print('--------')
            print(comp_img)
            print('--------')
            concat_data = np.expand_dims(np.append(mot, comp_img),axis=0)
            concat_data = concat_data[np.newaxis, :, :]
            print(concat_data)
            print('--------')
            print(normalized_range)
            print('--------')
            for j in range(concat_data.shape[2]):
                print(normalized_range[j])
                concat_data[:,:,j] = normalization(concat_data[:,:,j], normalized_range[j], after_range)
            print(concat_data)
            print('--------')


            print("MOT + IMAGE")
            print(concat_data)

            #sys.exit()

            result.append(concat_data)
            with open("./online_comp.csv", "w") as answer_file:
                for r in result:
                    for rr in r[0][0]:
                        answer_file.write(str(rr) + ",")
                    answer_file.write("\n")

            time.sleep(0.1)

            ##### RNN forward

            print("RNN Forward IN")
            hidden = None
            try:
                hidden = None
                cnt = 0
                        #x = Variable(torch.tensor(inp_rnn[1]))
                        #hidden = Variable(torch.tensor(hidden))
                if cnt > 15 or cnt==0:
                    print("online", cnt)
                    x = Variable(torch.tensor(concat_data))
                else:
                    print("closed", cnt)
                    x = y


                print("x shape: ", x.shape)


                y, hidden = model_LSTM.forward(x, hidden)
                cnt += 1
                print(y)
            except TypeError:
                print("xxx")
                


            print("Move ur5e IN")
            left_grip = "o"
            right_grip = "o"
            inp_motion = y

            for j in range(len(inp_motion[1][0][0][0:])):
                mot_n[j] = denormalization(inp_motion[1][0][0][j], normalized_range[j], after_range)


            m_jointPos = JARA_ARM.JointPos_var()
            m_LjointPos = JARA_ARM_LEFT.JointPos_var()

            for lst in range(len(6)):
                l_dic.update(mot_n[lst].data)
            for lst in range(len(6)):
                r_dic.update(mot_n[lst+len(left_rnn_data_order)+1].data)

            print("l_dic:", l_dic)
            print("r_dic:", r_dic)


            print("MOVE ROBOT START")
            ##TODO
            self._manipMiddle_R._ptr().movePTPJointAbs(m_jointPos)

            self._manipMiddle_L._ptr().movePTPJointAbs(m_LjointPos)

            time.sleep(1.0)
            print("MOVE ROBOT END")


            ##### gripper

            a = mot_n[6].to('cpu').detach().numpy()
            print("left gripper state: ", left_grip)

            if np.rint(a) == 0 and left_grip == "c":
                mot_n[6] = 100
                print("OPEN")
                left_grip = "o"

            elif np.rint(a) == 1 and left_grip == "o":
                #mot_n[6] = 0

                print("CLOSE")
                left_grip = "c"

            b = mot_n[13].to('cpu').detach().numpy()
            print("left gripper state: ", right_grip)

            if np.rint(b) == 0 and right_grip == "c":
                #mot_n[13] = 100
                self._manipMiddle._ptr().moveGripper(100)
                time.sleep(1.0)
                print("OPEN")
                right_grip = "o"

            elif np.rint(b) == 1 and right_grip == "o":
                #mot_n[13] = 0
                self._manipMiddle._ptr().moveGripper(0)
                time.sleep(1.0)
                print("CLOSE")
                right_grip = "c"

            print("MOVE Gripper Done")




            #JARA_ARM.CarPosWithElbow carPos

            carPos = JARA_ARM.CarPosWithElbow([[0,0,0,0],[0,0,0,0],[0,0,0,0]], 1.0, 1)
            carPos.carPos[0][0] = -c2;  carPos.carPos[0][1] = s2; carPos.carPos[0][2] =  0.0; carPos.carPos[0][3] = x;
            carPos.carPos[1][0] =  s2;  carPos.carPos[1][1] = c2; carPos.carPos[1][2] =  0.0; carPos.carPos[1][3] = y;
            carPos.carPos[2][0] =  0.0; carPos.carPos[2][1] = 0; carPos.carPos[2][2] = -1.0; carPos.carPos[2][3] = z;
            self._manipMiddle._ptr().movePTPCartesianAbs(carPos)

            time.sleep(1.0)

            carPos.carPos[2][3] = z_min
            self._manipMiddle._ptr().movePTPCartesianAbs(carPos)
            time.sleep(1.0)


            self._manipMiddle._ptr().moveGripper(10)#m_gripper_close_ratio*100)
            time.sleep(1.0)

            carPos.carPos[2][3] = z
            self._manipMiddle._ptr().movePTPCartesianAbs(carPos)
            time.sleep(1.0)

            self._manipMiddle._ptr().movePTPJointAbs([math.pi/2,0, math.pi/2, 0, math.pi/2, 0])
            time.sleep(3.0)
            self._manipMiddle._ptr().moveGripper(50)


        cv2.destroyAllWindows()
        return RTC.RTC_OK

    #   ##
    #   #
    #   # The aborting action when main logic error occurred.
    #   # former rtc_aborting_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onAborting(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The error action in ERROR state
    #   # former rtc_error_do()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onError(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The reset action that is invoked resetting
    #   # This is same but different the former rtc_init_entry()
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onReset(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The state update action that is invoked after onExecute() action
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #

    #   #
    #def onStateUpdate(self, ec_id):
    #
    #   return RTC.RTC_OK

    #   ##
    #   #
    #   # The action that is invoked when execution context's rate is changed
    #   # no corresponding operation exists in OpenRTm-aist-0.2.0
    #   #
    #   # @param ec_id target ExecutionContext Id
    #   #
    #   # @return RTC::ReturnCode_t
    #   #
    #   #
    #def onRateChanged(self, ec_id):
    #
    #   return RTC.RTC_OK




def ChouchoGeneratorInit(manager):
    profile = OpenRTM_aist.Properties(defaults_str=armimagepredictor_keras_spec)
    manager.registerFactory(profile,
                            ChouchoGenerator,
                            OpenRTM_aist.Delete)

def MyModuleInit(manager):
    ChouchoGeneratorInit(manager)

    # Create a component
    comp = manager.createComponent("ChouchoGenerator")

def main():
    mgr = OpenRTM_aist.Manager.init(sys.argv)
    mgr.setModuleInitProc(MyModuleInit)
    mgr.activateManager()
    mgr.runManager()

if __name__ == "__main__":
    main()
