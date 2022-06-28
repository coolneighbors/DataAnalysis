# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:35:23 2022

@author: Noah Schapera
"""

import csv
import json 
import copy

def stringToJson(string):
    res = json.loads(string)
    return res

def csvToDict(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return csv_reader

def parseTargets(subject_file_string, class_file_string):
    
    sub_file = open(subject_file_string, mode='r')

    class_file = open(class_file_string, mode='r')
    
    subjects = csv.DictReader(sub_file)
    classifications = csv.DictReader(class_file)
    classifications_list = copy.copy(list(classifications))
    
    for sub in subjects:
        subBool = None
        subID=sub["subject_id"]
        
        metadata_json = stringToJson(sub["metadata"])
        
        subClass = metadata_json["#R/F"]
        
        if subClass == "R":
            subBool = True
        else:
            subBool = False
            
            
        correctClass=0
        incorrectClass=0
        
        
        for cl in classifications_list:           
            classID = cl["subject_ids"]
            
            if classID == subID:
                
                anno = stringToJson(cl["annotations"])
                
                if anno[0]["value"] == "Yes":
                    clBool = True
                else:
                    clBool = False
                    
                if subBool == clBool:
                    correctClass += 1
                else:
                    incorrectClass += 1
                
        print(f"Subject {subID} was classified correctly {correctClass} times and incorrect {incorrectClass} times")
    
    
    class_file.close()
    sub_file.close()
                
parseTargets("byw-cn-test-project-subjects.csv","byw-cn-test-project-classifications.csv")
        
        
        
    
        

    