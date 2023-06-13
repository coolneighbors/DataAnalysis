# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 13:35:23 2022

@author: Noah Schapera
"""

import csv
import json 
import copy

class Parser:
    def __init__(self, subject_file_string_in, class_file_string_in):
        
        self.class_file_string = class_file_string_in
        self.subject_file_string = subject_file_string_in
        
        self.targets_classificationsToList()

    def stringToJson(self, string):
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
    
    def getUsersForSubject(self, subID):
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
    
    def classifySubjectByThreshold(self, threshold, subID):
        '''
        iterates through classification list to find matches to a particular subect. Counts how many times each matching classification marks the subject as a mover. If its above a threshold, returns true

        Parameters
        ----------
        threshold : integer
            classification threshold.
        subID : string
            ID of a particular subject that is being matched.

        Returns
        -------
        bool
            true or false, has the number of classifications passed the threshold.
        yesCounter: int
            number of times the classifications were "Yes".

        '''
        yesCounter=0
        for cl in self.classifications_list:
            classID=cl["subject_ids"]

            
            if classID == subID:
                anno = self.stringToJson(cl["annotations"])
                if anno[0]["value"] == "Yes":
                    yesCounter+=1
        if yesCounter >= threshold:
            return True, yesCounter
        else:
            return False, 0

    def classifyAllSubjects(self, threshold):
        '''
        Iterates through all subjects and classifications, determines how many times each subject was classified as a mover. If the number of classifications is above
        a set threshold, returns the subject as a likely mover

        Parameters
        ----------
        threshold : Integer
            Number of times a subject must be classified as "yes" for it to be counted as a mover

        Returns
        -------
        movers : Nx2 list
            index [i][0] is the subject ID of the mover
            idex [i][1] is the number of times the mover was classified as "Yes"

        '''
        movers=[]

        for sub in self.subjects_list:
            subID=sub["subject_id"]
            isMover, numYes = self.classifySubjectByThreshold(threshold,subID)
            if isMover:
                mov=[subID,numYes]
                movers.append(mov)
        return movers
        
    
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
    #P.printAccuracy()
    ms= P.classifyAllSubjects(threshold=2)
    us = P.getUniqueUsers()
    for m in ms:
        print(m[0] + " was classified as a mover " + str(m[1]) + " times")
    for u in us:
        print(u)
    P.printAccuracy()