import glob
import os.path
import json

class AzureTemplateLoader():
    """ Simple Azure Resource Manager template loader:
        - given a path, recurse through it and look for the presence of metadata.json file to extract "training prompt"
        - in that same path, look for azuredeploy.json file - this will the "desired output" of our models
    """
    def __init__(self, path_to_arm_templates):
        """ Load training data from the specified path.

        @param path_to_arm_templates (string): Path to ARM templates
                """
        template_count = 0
        prompt_prefix = "\n\n###\n\n"
        prompt_suffix = "\n\n===\n\n"
        completion_prefix = " "
        completion_suffix = " END"

        for file_name in glob.iglob(path_to_arm_templates + '/**/metadata.json', recursive=True):
            # extract Summary and (later) Description from metadata - this will be our prompt
            metadata_file = open(file_name)
            metadata_json = json.load(metadata_file)
            metadata_file.close()
            metadata_summary = prompt_prefix + metadata_json['summary'] + prompt_suffix

            # the resource we are looking for is Microsoft.* which should be a part of the path (filename)
            resource_start = file_name.lower().find("microsoft")
            resource_end = file_name.lower().find("/", resource_start)
            resource_name = file_name[resource_start: resource_end]


            if len(resource_name) > 0:
                azure_deploy_path = file_name.replace("metadata", "azuredeploy")
                azuredeploy_exists = os.path.exists(azure_deploy_path)
                if azuredeploy_exists:
                    # find the resource_name element and encode it as a string - this will be our target contents to generate
                    #  initial experiment on the full template have been extremely poor, so scaling down to specific resources only
                    print(azure_deploy_path)
                    arm_template_file = open(azure_deploy_path)
                    arm_template_json =  json.load(arm_template_file)
                    arm_template_file.close()
                    arm_template_string = ""

                    all_resources = arm_template_json['resources']
                    for resource in all_resources:
                        if resource_name.lower() in str(resource['type']).lower():
                            arm_template_string = arm_template_string + str(resource).lower()

                    completion_arm = completion_prefix + arm_template_string + completion_suffix

                    # every 50th template will be our test set
                    if template_count % 50 == 0:
                        data_line = { "prompt" : metadata_summary, "truth": completion_arm }
                        data_line_string = json.dumps(data_line) + "\n"
                        training_file = open("test.json", "a")
                        write_it = training_file.write(data_line_string)
                        
                    else:
                        # goal is to generate file for fine-tuning in the form
                        #  {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
                        data_line = { "prompt" : metadata_summary, "completion": completion_arm }
                        data_line_string = json.dumps(data_line) + "\n"
                        training_file = open("train.json", "a")
                        write_it = training_file.write(data_line_string)
                    
                    training_file.close()
                    template_count += 1

        print("total files " + str(template_count))