# DataAnalysis

A simple parsing object which allows users to extract data from zooniverse classifications. Requires a classifications and subject file as exported from zooniverse.

## Dependencies
csv, json, and copy libraries (use pip or similar for installation)

## Usage
Parser object(subject filename, classification filename0
Current functionality: 
	* Identifies users who contributed to a particular subjects classification
	* Identifies all unique users who contributed to a project

testParser, subclass of Parser
Designed to work with 30 subject classification test for the Backyard Worlds: Cool Neighbors development
Functionality: 
	* printAccuracy -- Compares hidden subject #R/F metadata to user classifications to determine accuracy for each subject. Prints results to console.

