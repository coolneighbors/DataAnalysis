# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:35:23 2022

@author: Noah Schapera
"""

import csv
import json 

def stringToJson(string):
    res = json.loads(string)
    return res

def csvToDict(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        return csv_reader

def parseTargets(subject_file_string, class_file_string):
    
    with open(subject_file_string, mode='r') as sub_file:
        subjects = csv.DictReader(sub_file)
        with open(class_file_string, mode='r') as class_file:
            classifications = csv.DictReader(class_file)
            for sub in subjects:
                subID=sub["subject_id"]
                metadata_json = stringToJson(sub["metadata"])
                subClass = metadata_json["#R/F"]
                if subClass == "Y":
                    subBool = True
                else:
                    subBool = False
                correctClass=0
                incorrectClass=0
                for cl in classifications:
                    classID = cl["subject_ids"]
                    print("sub id: " + subID)
                    print("class id: " + classID)
                    if classID == subID:
                        print("Match")
                        anno = stringToJson(cl["annotations"])
                        if anno[0]["value"] == "Yes":
                            clBool = False
                        else:
                            clBool = True
                            
                        if subBool == clBool:
                            correctClass += 1
                        else:
                            incorrectClass += 1
                        
                print(f"Subject {subID} was classified correctly {correctClass} times and incorrect {incorrectClass} times")
                
parseTargets("byw-cn-test-project-subjects.csv","byw-cn-test-project-classifications.csv")
        
        
        
    
        

    