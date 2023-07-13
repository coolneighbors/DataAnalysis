import csv

def in_targets_file(TARGET_ID):
    TARGET_ID = int(TARGET_ID)
    target_filename = ""
    with open(target_filename, newline='') as targets_file:
        reader = csv.DictReader(targets_file)
        for row in reader:
            if(TARGET_ID == int(row["TARGET ID"])):
                return True
    return False

def not_in_targets_file(TARGET_ID):
    TARGET_ID = int(TARGET_ID)
    target_filename = ""
    with open(target_filename, newline='') as targets_file:
        reader = csv.DictReader(targets_file)
        for row in reader:
            if(TARGET_ID == int(row["TARGET ID"])):
                return False
    return True

def in_workflow(workflow_id):
    workflow_id = int(workflow_id)
    if(workflow_id == 0):
        return True
    else:
        return False

def in_subject_set(subject_set_id):
    subject_set_id = int(subject_set_id)
    if(subject_set_id == 0):
        return True
    else:
        return False

def all_subjects():
    return True

