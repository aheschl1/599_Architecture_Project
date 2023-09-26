import glob
import xlsxwriter as xlsx
import os
import json

def extract_decimal(line:str)->float:
    section = line.split(' ')[-2]
    section = section[1:-1]
    return float(section)

def analyze_log(log_name:str)->dict:
    best_scores = None
    best_epochs = None
    params = 0
    times = []
    kwargs = ""

    with open(log_name) as log:
        epoch = 0
        for line in log.readlines():
            if "'group':" in line:
                kwargs = ' '.join(line.split(' ')[2:])
            if "Model parameters" in line:
                try:
                    params = int(str(line.split(' ')[-2]).strip())
                except:
                    params = 32753075
            if 'Epoch time' in line:
                times.append(float(line.split(' ')[4]))
            if 'Epoch' in line and not 'time' in line and not '200_' in line:
                epoch = int(line.split(' ')[3])
            if 'Pseudo dice' in line:
                # if extract_decimal(line) > best_dice:
                #     best_epoch = epoch
                #     best_dice = extract_decimal(line)
                scores = eval(line[line.find('['):])
                if best_scores == None:
                    best_scores = scores
                    best_epochs = [epoch, epoch, epoch]
                else:
                    for i, data in enumerate(scores):
                        if best_scores[i] < data:
                            best_scores[i] = data
                            best_epochs[i] = epoch
    
    return kwargs[kwargs.find(':') + 1 : kwargs.find(',')].replace("'", " ").strip() if kwargs!="" else "nnUNet", {
        "best_scores" : best_scores,
        "best_epochs" : best_epochs,
        "name" : log_name.split('/')[-1].split('.')[-2].split('__')[0],
        "params" : params,
        "time" : sum(times)/(len(times)+0.00001),
        "kwargs" : kwargs,
        "path":log_name
    }

def main(folder:str):

    log_paths = glob.glob(f'/home/andrewheschl/Documents/logs/{folder}/**/*.txt', recursive=True)

    info = {} 

    for log in log_paths:
        group, data = analyze_log(log)
        if not group in info.keys():
            info[group] = []
        info[group].append(data)

    workbook = xlsx.Workbook(f'/home/andrewheschl/Documents/logs/Result/{folder}_logs.xlsx')
    my_format = workbook.add_format()
    my_format.set_align('right')
    
    kjkj = 0
    for group, logs in info.items():
        worksheet = workbook.add_worksheet(group if group != "dsc" else "dsc" + str(kjkj))
        kjkj += 1
        worksheet.set_column('A:XFD', None, my_format)
        worksheet.set_row(0, None, cell_format=workbook.add_format({"bold":True}))

        if group != "nnUNet":
            kwargs = json.loads(logs[0]['kwargs'].replace("'", '"').replace("True", "true").replace("False", "false"))
            if not "Loss" in kwargs:
                kwargs["Loss"] = "Default"
            kwargs.pop('group')
        
        if logs[0]['best_scores'] != None:
            num_classes = len(logs[0]['best_scores'])
        else:
            print(f"Fail on reading from {group}. No best scores in item 0. Skipping group and deleting log.")
            os.remove(logs[0]['path'])
            continue

        worksheet.write(0, 0, 'File')
        worksheet.write(0, 1, 'Average Time')
        worksheet.write(0, 2, 'Parameters e-6')

        epoch_column = 0
        column_to_write = 3
        if group != "nnUNet":
            for c, key in enumerate(kwargs.keys(), column_to_write):
                worksheet.write(0, c, key.title())
                column_to_write += 1
        
        worksheet.write(0, column_to_write, "Full Kidney")
        column_to_write += 1

        if num_classes == 3:
            worksheet.write(0, column_to_write, "Both Masses")
            worksheet.write(0, column_to_write+1, "Only Tumour")
            column_to_write += 2
        
        worksheet.write(0, column_to_write, "Epochs")
        epoch_column = column_to_write

        for r, data in enumerate(logs, 1):
            worksheet.write(r, 0, data['name'])
            worksheet.write(r, 1, round(data['time'], 2))
            worksheet.write(r, 2, round(data['params']/1e6, 3))

            column_to_write = 3
            if group != "nnUNet":
                kwargs = json.loads(data['kwargs'].replace("'", '"').replace("True", "true").replace("False", "false"))
                if not "Loss" in kwargs:
                    kwargs["Loss"] = "Default"
                kwargs.pop('group')
                for c, key in enumerate(kwargs.keys(), column_to_write):
                    worksheet.write(r, c, kwargs[key])
                    column_to_write = c+1
            
            if data['best_scores'] != None:
                for score in data['best_scores']:
                    worksheet.write(r, column_to_write, score)
                    column_to_write += 1
            else:
                print(f"Failed reading from an element of {group}. No best scores in this file. Skipping and deleting file.")
                os.remove(data['path'])
                continue

            worksheet.write(r, epoch_column, str(data['best_epochs']))
                        
        worksheet.autofit()
        print(f"Group {group}: Success")

    workbook.close()

if __name__ == "__main__":
    main(str(input("Folder name: ")))