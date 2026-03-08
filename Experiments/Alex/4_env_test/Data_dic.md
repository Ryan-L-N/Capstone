
Data Dictionary Guidelines
<student and mentor names>
<submission date>
<version>
Purpose
The purpose of a data dictionary is to define all the data that has been collected in such a way that a non-team member can clearly understand it. This is very important documentation when handing off a project. This is probably best recorded in a spreadsheet.
Instructions
Create a spreadsheet <Capstone_Project_Name>_Data_Dictionary with the following:

Identify all the data variables that were recorded.
For each data variable, do the following in a spreadsheet:
Provide the name of the variable as it is used.
Provide a short descriptive name if the original name is not obvious.
Provide a short definition of the variable’s purpose.
Provide a description of the type of data (numeric, character, string, Boolean, etc.). This could be the data type used in the database or programming language.
Provide the range of acceptable values:
Minimum to maximum of numeric values.
Provide minimum and maximum lengths of strings.
Provide a column that states if the data is required for every record or optional.
Provide a column with any notes that may be necessary for interpretation, important exceptions, etc.
Check the definitions thoroughly.  Make sure multiple team members review the dictionary.
Complete the reflection and the appendix (if needed).

Reflection
Use this section to write a brief description of what you learned in the process of making this document: what will I do differently next time, what I learned from working in a team, etc. Then, reflect on the decision-making process when making this document. 
Appendix
This section contains any additional information you’d like to preserve in this document for context. For example, consider having a Glossary, or any additional materials you discovered or created in the process of making this document.

Changes To Previous Deliverables
Use this section to outline any changes you had to make in your previous documents.

—------------------------------------------
Writing, Style, and Formatting
Overall writing requirements
Make sure your writing is brief and easy to understand. The entire document (not including references and appendix) should be about 2 to 5 pages. Please take time to edit and proofread your work before submitting it.



Exploratory Data Analysis Template
<student and mentor names>
<submission date>
<version>
Purpose
The purpose of exploratory data analysis (EDA) is to understand your data for project analysis, iteration, and evaluation.
Instructions
In a document file called <Capstone_Project_Name>_Exploratory_Data_Anaysis, provide the following sections.
Examine the Data Distributions
Plot the variables of your data using histograms, bar charts, or whatever is appropriate.
Confirm that the distributions have reasonable shapes and values..
Identify any distributions that show unexpected behavior.
If there’s an unexpected shape, identify potential reasons for the problem.
Identify if there are any issues that could render the data unusable. 
For categorical data, report the counts.
If the categorical data contains very unequal groups, identify if this is expected or may be problematic for later (statistical) analyses.
Check for Missing Data
Identify if any data is missing.
If data is missing, identify if there could be a systematic cause for the missing data.
If data is missing, identify whether the missing data should be ignored, if missing data records should be dropped, or if missing data should be imputed.
If imputing data, identify an appropriate method for imputation such as the average, median, or another calculation. Make sure to check appropriate literature or consult with experts to know which imputation techniques are appropriate..
Check for Outliers
Use a box-and-whisker plot or other appropriate technique to identify outliers.
If outliers are present, identify if they represent legitimate data or a mistake.
If outliers are present, determine if they should be included or excluded. If the outliers are excluded, give an explanation for why they should be eliminated.
Report Descriptive Statistics
Check and report the minimum, mean, median, mode, and maximum.
If the data is expected to be normally distributed, consider using appropriate tests for normality.
If the data is normally distributed, report the standard deviation with the mean.
Report the counts of nominal or categorical variables.
Further Considerations
Use scatterplots to check variable relationships.
Calculate variable correlations to identify or check meaningful associations.
Consider transforming non-normal data into normal data if it aids legitimate comparison.
Consider using other visualizations such as Sankey graphs, violin charts, etc.
Consider using any metrics or visualizations that tell a story about the data and help you and your stakeholders evaluate the state of the project.
Complete the reflection and the appendix (if needed).

Reflection
Use this section to write a brief description of what you learned in the process of making this document: what will I do differently next time, what I learned from working in a team, etc. Then, reflect on the decision-making process when making this document. 
Appendix
This section contains any additional information you’d like to preserve in this document for context. For example, consider having a Glossary, or any additional materials you discovered or created in the process of making this document.

Changes To Previous Deliverables
Use this section to outline any changes you had to make in your previous documents.

Writing, Style, and Formatting
Overall writing requirements
Make sure your writing is brief and easy to understand. The entire document (not including references and appendix) should be about 2 to 5 pages. Please take time to edit and proofread your work before submitting it.



Data Schema Guidelines
<student and mentor names>
<submission date>
<version>
Purpose
The purpose of a data schema template is to define an implementation of a data dictionary. Data schemas are usually implemented in JSON, XML, or some other markup format.  The data schema can be validated using a validation tool appropriate to the format (and programming language if appropriate).

Instructions
Create a JSON file (or other markup language format if appropriate)  <Capstone_Project_Name>_Data_Schema with the following:

Complete the Data Dictionary Template to document all data variables.
Select a format for the schema. Make sure to check with the stakeholders to see if a specific format is required.
Translate each variable in the data dictionary into the appropriate syntax of the target format.
Identify how columns of the dictionary should be translated into attributes.
Find an appropriate validator for the target format and target programming/scripting language if appropriate.
Validate the template and adjust as necessary.
Have team members review and test the schema.
Complete the reflection and the appendix (if needed).

Reflection
Use this section to write a brief description of what you learned in the process of making this document: what will I do differently next time, what I learned from working in a team, etc. Then, reflect on the decision-making process when making this document. 
Appendix
This section contains any additional information you’d like to preserve in this document for context. For example, consider having a Glossary, or any additional materials you discovered or created in the process of making this document.

Changes To Previous Deliverables
Use this section to outline any changes you had to make in your previous documents.

—------------------------------------------
Writing, Style, and Formatting
Overall writing requirements
Make sure your writing is brief and easy to understand. The entire document (not including references and appendix) should be about 2 to 5 pages. Please take time to edit and proofread your work before submitting it.

Model Documentation
<student and mentor names>
<submission date>
<version>

Overview
Briefly introduce the model in non-technical terms, focusing on its main characteristics and typical use cases. You can briefly touch on how the typical use cases relate to your particular use case in order to provide motivation for using the chosen model. Include reasoning behind selecting this model for the task.
Specifications
Describe the model architecture in technical terms. For example, if the model is a neural network then you want to provide information about the number layers, number of nodes in each layer and activation functions. Refer to the model documentation and make sure to include pointers to the implementation that you used (e.g., GitHub repo or Python library on PyPi). Include diagrams of model architecture if appropriate.
Model Run
Use this section to describe anything related to running the model. Focus especially on the platform where you run the model, and model hardware requirements. Provide some information about the cost of running the model.
Training
OPTIONAL Describe how you trained the model in case there was training involved (i.e., you are not using an off-the-shelf pre-trained model). Provide information about the dataset used for training. Provide the parameters of the training in a way that someone that is not part of your team could replicate the procedure.
Evaluation
Describe and justify the evaluation procedures you have chosen. Provide the results of the evaluation, your analysis of the results, and your reflection on the process.
References
[C&I, 2016] Complicated & Important, If you have many references, they should go into a bibliography appendix such as this one!, Proceedings of Whatever, 67-98, 2016.

[Also-Important, 2016] Also-Important et al., How you format these individual references is not that important as long as it is consistent, Journal of Meaningful Studies, Vol. 16, 112-120, 2016.

References should follow the IEEE standards. Follow this guideline to cite references. 
Reflection
Use this section to write a brief description of what you learned in the process of making this document: what will I do differently next time, what I learned from working in a team, etc. Then, reflect on the decision-making process when making this document. Reflection points to consider:
…
…
Appendix
This section contains any additional information you’d like to preserve in this document for context. For example, consider having a Glossary, or any additional materials you discovered or created in the process of making this document.

Changes To Previous Deliverables
Use this section to outline any changes you had to make in your previous documents.

—------------------------------------------
Writing, Style, and Formatting
Overall writing requirements
Make sure your writing is brief and easy to understand. 1 to 3 paragraphs per section. The entire document (not including references and appendix) should be about 2 to 5 pages. Please take time to edit and proofread your work before submitting it.
As always, if you produce subsections
Make sure that you use the proper sub-heading style.
The same goes for Sub-sub-headings
This is important because the documents you produce may be read by people who are not close collaborators and for whom a well-structured document is helpful to understand things. Also, remember to cite the things you use [C&I, 2016].
