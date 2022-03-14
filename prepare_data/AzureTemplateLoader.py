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
        available_templates = 0
        prompt_prefix = "\n\n###\n\n"
        prompt_suffix = "\n\n===\n\n"
        completion_prefix = " "
        completion_suffix = " END"
        dict_original_infra = {}
        
        for file_name in glob.iglob(path_to_arm_templates + '/**/metadata.json', recursive=True):
            available_templates += 1

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
                    dict_original_infra[resource_name] = metadata_summary
                    
                    # every 10th template will be our test set
                    if template_count % 10 == 0:
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

        print("total files " + str(template_count) + ", available files = " + str(available_templates))
        self.get_more_data(path_to_arm_templates, dict_original_infra)

    def get_more_data(self, path_to_arm_templates, dict_original_infra):
        # in pure template definitions, there just isn't enough data for training
        #  so use a bit of creativity here: since templates are very likely to contain elements not of just main type
        #   i.e., Microsoft.Compute/VM, but also Microsoft.Networking/network, etc. get those elemets as additional 
        #   "syndicated" data into the dataset
        prompt_prefix = "\n\n###\n\n"
        prompt_suffix = "\n\n===\n\n"
        completion_prefix = " "
        completion_suffix = " END"
        additional_resources = 0
        
        print("\n\n========GETTING MORE DATA======")
        for file_name in glob.iglob(path_to_arm_templates + '/**/azuredeploy.json', recursive=True):
            # the resource we are looking for is Microsoft.* which should be a part of the path (filename)
            file_resource_start = file_name.lower().find("microsoft")
            file_resource_end = file_name.lower().find("/", file_resource_start)
            file_resource_name = file_name[file_resource_start: file_resource_end]

            for first_pass_resource in dict_original_infra:
                # skip test folder
                if file_name.find("/test/") < 0:
                    if first_pass_resource != file_resource_name:
                        print(file_name)
                        arm_template_file = open(file_name)
                        arm_template_json =  json.load(arm_template_file)
                        arm_template_file.close()
                        arm_template_string = ""

                        all_resources = arm_template_json['resources']
                        for resource in all_resources:
                            if first_pass_resource.lower() in str(resource['type']).lower():
                                arm_template_string = arm_template_string + str(resource).lower()
                                completion_arm = completion_prefix + arm_template_string + completion_suffix

                                # every 10th template will be our test set
                                if additional_resources % 10 == 0:
                                    data_line = { "prompt" : dict_original_infra[first_pass_resource], "truth": completion_arm }
                                    data_line_string = json.dumps(data_line) + "\n"
                                    training_file = open("test.json", "a")
                                    write_it = training_file.write(data_line_string)
                                    
                                else:
                                    # goal is to generate file for fine-tuning in the form
                                    #  {"prompt": "<prompt text>", "completion": "<ideal generated text>"}
                                    data_line = { "prompt" : dict_original_infra[first_pass_resource], "completion": completion_arm }
                                    data_line_string = json.dumps(data_line) + "\n"
                                    training_file = open("train.json", "a")
                                    write_it = training_file.write(data_line_string)
                                
                                training_file.close()
                                additional_resources += 1

        print("total additional resources found " + str(additional_resources))