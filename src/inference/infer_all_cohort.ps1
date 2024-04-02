# General info
$model_sign_path = "C:\Users\inserm\Documents\histo_sign\dataset\all_model_path.npy"
$model_tum_path = "C:\Users\inserm\Documents\histo_sign\trainings\tumors\2024-03-25_14-43-57\model.pth"


# MDN 
$export_dir_mdn = "C:\Users\inserm\Documents\histo_sign\dataset\inference_mdn"
$features_dir_mdn = "C:\Users\inserm\Documents\histo_sign\dataset\features_mdn_224_ctranspath"
$summary_data = "C:\Users\inserm\Documents\histo_sign\dataset\mdn_summary_vst.csv"

# Prodige24
$export_dir_prodige24 = "C:\Users\inserm\Documents\histo_sign\dataset\inference_prodige24" 
$features_dir_prodige24 = "C:\Users\inserm\Documents\histo_sign\dataset\features_p24_224_ctranspath"
$summary_data_prodige24 = "C:\Users\inserm\Documents\histo_sign\dataset\p24_summary_vst.csv"

# Multicentric
$export_dir_multicentric = "C:\Users\inserm\Documents\histo_sign\dataset\inference_panc"
$features_dir_multicentric = "C:\Users\inserm\Documents\histo_sign\dataset\features_panc_224_ctranspath"
$summary_data_multicentric = "C:\Users\inserm\Documents\histo_sign\dataset\panc_summary_vst.csv"

python "C:\Users\inserm\Documents\histo_sign\src\inference\infererence_cohort.py" `
--export_dir $export_dir_mdn `
--model_sign_path $model_sign_path `
--model_tum_path $model_tum_path `
--features_dir $features_dir_mdn `
--summary_data $summary_data

python "C:\Users\inserm\Documents\histo_sign\src\inference\infererence_cohort.py" `
--export_dir $export_dir_prodige24 `
--model_sign_path $model_sign_path `
--model_tum_path $model_tum_path `
--features_dir $features_dir_prodige24 `
--summary_data $summary_data_prodige24

python "C:\Users\inserm\Documents\histo_sign\src\inference\infererence_cohort.py" `
--export_dir $export_dir_multicentric `
--model_sign_path $model_sign_path `
--model_tum_path $model_tum_path `
--features_dir $features_dir_multicentric `
--summary_data $summary_data_multicentric
