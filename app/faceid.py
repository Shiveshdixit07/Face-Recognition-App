from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class camApp(App):
    def build(self):
        self.web_cam=Image(size_hint=(1,.8))
        self.button=Button(text='Verify',on_press=self.verify,size_hint=(1,.1))
        self.verification_label=Label(text='Verification Uninitiated',size_hint=(1,.1))

        layout=BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('model.h5', custom_objects={'L1Dist':L1Dist})


        self.capture=cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_cam, 1.0/33.0)
        return layout
        
    def update_cam(self,*args):
        ret,frame=self.capture.read()
        frame=frame[130:130+250,250:250+250,: ]

        buf=cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocessing(self,file_path):
        byte_image=tf.io.read_file(file_path)
        img=tf.io.decode_jpeg(byte_image)
        img=tf.image.resize(img,(100,100))
        img=img/255.0
        return img    

    def verify(self,*args):
        verification_threshold=0.7
        detection_threshold=0.5

        save_path=os.path.join('app_data','input_images','input_image.jpg')
        ret,frame=self.capture.read()
        frame=frame[130:130+250,250:250+250,: ]
        cv2.imwrite(save_path,frame)


        results=[]
        for img in os.listdir(os.path.join('app_data','verification_images')):
            input_image=self.preprocessing(os.path.join('app_data','input_images','input_image.jpg'))
            validation_image=self.preprocessing(os.path.join('app_data','verification_images',img))
        
            result=self.model.predict(list(np.expand_dims([input_image,validation_image],axis=1)))
            results.append(result)
    
        detection=np.sum(np.array(results)>detection_threshold)

        verification=detection/len(os.listdir(os.path.join('app_data','verification_images')))
        verified=(verification>verification_threshold)

        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)


        return results,verified   


if __name__ == '__main__':
    camApp().run()   
