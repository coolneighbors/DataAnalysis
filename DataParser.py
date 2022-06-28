# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:35:23 2022

@author: Noah Schapera
"""

import csv
import json 
import copy

class Parser:
    def __init__(self, subject_file_string_in,class_file_string_in):
        
        self.class_file_string = class_file_string_in
        self.subject_file_string = subject_file_string_in
        
        self.targets_classificationsToList()
        
        
        
    def stringToJson(self,string):
        '''
        Converts a string (metadata) to a json

        Parameters
        ----------
        string : String
            Any dictionary formatted as a string.

        Returns
        -------
        res : json object (dictionary)
            A dictionary.

        '''
        res = json.loads(string)
        return res
    
    def getUsersForSubject(self,subID):
        '''
        Returns users who contributed to a subject classification

        Parameters
        ----------
        subID : string
            ID of a particular subject of interest.

        Returns
        -------
        users : list, strings
            list of users who contributed to a subject classification.

        '''
        users = []
        for cl in self.classifications_list:
            classID = cl["subject_ids"]
            
            if subID == classID:
                users.append(cl["user_name"])
        return users
        
    
    def getUniqueUsers(self):
        '''
        returns all unique users who contributed to a project

        Returns
        -------
        uniqueUsers : list, string
            list of all users who contributed to the project.

        '''
        uniqueUsers = []
        for cl in self.classifications_list:
            if len(uniqueUsers) == 0:
                uniqueUsers.append(cl["user_name"])
            else:
                isUnique = True
                for u in uniqueUsers:
                    if cl["user_name"] == u:
                        isUnique = False
                if isUnique:
                    uniqueUsers.append(cl["user_name"])
        return uniqueUsers
    

    def targets_classificationsToList(self):
    
        sub_file = open(self.subject_file_string, mode='r')
    
        class_file = open(self.class_file_string, mode='r')
        
        self.subjects_list = copy.deepcopy(list(csv.DictReader(sub_file)))
        self.classifications_list = copy.deepcopy(list(csv.DictReader(class_file)))
        
        class_file.close()
        sub_file.close()
        
class testParser(Parser):
    '''
    Child class of parser, for Cool Neighbors test
    '''
    def __init__(self, subject_file_string_in,class_file_string_in):
        super().__init__(subject_file_string_in,class_file_string_in)
        
    def printAccuracy(self):
        '''
        Compares known classifications (R/F) within the subjects hidden metadata to users classifications to determine accuracy.
        
        Only useful for this particular test
        Returns
        -------
        None.

        '''
        for sub in self.subjects_list:
            subBool = None
            subID=sub["subject_id"]
            metadata_json = self.stringToJson(sub["metadata"])
            subClass = metadata_json["#R/F"]
            
            if subClass == "R":
                subBool = True
            else:
                subBool = False
                
                
            correctClass=0
            incorrectClass=0
            
            
            for cl in self.classifications_list:           
                classID = cl["subject_ids"]
                
                if classID == subID:
                    
                    anno = self.stringToJson(cl["annotations"])
                    
                    if anno[0]["value"] == "Yes":
                        clBool = True
                    else:
                        clBool = False
                        
                    if subBool == clBool:
                        correctClass += 1
                    else:
                        incorrectClass += 1
                    
            print(f"Subject {subID} was classified correctly {correctClass} times and incorrect {incorrectClass} times")
        
        
        
        
        

if __name__ == "__main__":
    P = testParser("byw-cn-test-project-subjects.csv","byw-cn-test-project-classifications.csv")
    P.printAccuracy()
    us = P.getUniqueUsers()
    for u in us:
        print(u)
    
    
    
        
        
    
        

    