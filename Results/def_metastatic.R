
df = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/TCGA_survival_data_clean.txt")


unique(df$clinical_stage )
df$clinical_stage = as.character(df$clinical_stage)
df$ajcc_pathologic_tumor_stage = as.character(df$ajcc_pathologic_tumor_stage)
df$stage=case_when(
  df$clinical_stage  == "Stage I" ~ "Stage I",
  df$clinical_stage == "Stage IB2" ~ "Stage I",
  df$clinical_stage == "Stage IIIB" ~ "Stage III",
  df$clinical_stage == "Stage IB" ~ "Stage I",
  df$clinical_stage == "Stage IB1" ~ "Stage I",
  df$clinical_stage == "Stage IIB" ~ "Stage II",
  df$clinical_stage == "Stage IIA" ~ "Stage II",
  df$clinical_stage == "Stage IVA" ~ "Stage IV",
  df$clinical_stage == "Stage IVB" ~ "Stage IV",
  df$clinical_stage == "Stage IIA2" ~ "Stage II",
  df$clinical_stage == "Stage II" ~ "Stage II",
  df$clinical_stage == "Stage IIA1" ~ "Stage II",
  df$clinical_stage == "Stage IA2" ~ "Stage I",
  df$clinical_stage == "Stage III" ~ "Stage III",
  df$clinical_stage == "Stage IA" ~ "Stage I",
  df$clinical_stage == "Stage IA1" ~ "Stage I",
  df$clinical_stage == "Stage I" ~ "Stage I",
  df$clinical_stage == "Stage IIIA" ~ "Stage III",
  df$clinical_stage == "Stage IV" ~ "Stage IV",
  df$clinical_stage == "Stage IIIC" ~ "Stage III",
  df$clinical_stage == "Stage IVC" ~ "Stage IV",
  df$clinical_stage == "Stage IIC" ~ "Stage II",
  df$clinical_stage == "Stage IC" ~ "Stage I",
  df$clinical_stage == "Stage IS" ~ "Stage I",
  df$clinical_stage == "I" ~ "Stage I",
  df$clinical_stage == "II" ~ "Stage II",
  df$clinical_stage == "III" ~ "Stage III",
  df$clinical_stage == "IIa" ~ "Stage II",
  df$clinical_stage == "IIb" ~ "Stage II",
  df$clinical_stage == "IIc" ~ "Stage II"
)
df$stage2=case_when(
  df$ajcc_pathologic_tumor_stage  == "Stage I" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IB2" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IIIB" ~ "Stage III",
  df$ajcc_pathologic_tumor_stage == "Stage IB" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IB1" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IIB" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage IIA" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage IVA" ~ "Stage IV",
  df$ajcc_pathologic_tumor_stage == "Stage IVB" ~ "Stage IV",
  df$ajcc_pathologic_tumor_stage == "Stage IIA2" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage II" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage IIA1" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage IA2" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage III" ~ "Stage III",
  df$ajcc_pathologic_tumor_stage == "Stage IA" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IA1" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage I" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IIIA" ~ "Stage III",
  df$ajcc_pathologic_tumor_stage == "Stage IV" ~ "Stage IV",
  df$ajcc_pathologic_tumor_stage == "Stage IIIC" ~ "Stage III",
  df$ajcc_pathologic_tumor_stage == "Stage IVC" ~ "Stage IV",
  df$ajcc_pathologic_tumor_stage == "Stage IIC" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "Stage IC" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "Stage IS" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "I" ~ "Stage I",
  df$ajcc_pathologic_tumor_stage == "II" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "III" ~ "Stage III",
  df$ajcc_pathologic_tumor_stage == "IIa" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "IIb" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "IIc" ~ "Stage II"
)
df$final_stage = unlist(apply(df, 1, function(x){
  if (!is.na(x["stage"])){
    x["stage"]
  } else if (!is.na(x["stage2"])){
    x["stage2"]
  } else {
    NA
  }
}))

All_TCGA_CIN_measures = readRDS("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/All_TCGA_CIN_measures.rds")
df$metastatic_one_two_three = ifelse(df$final_stage == "Stage IV", 1, 0)
table(df$metastatic_one_two_three)

df3 = merge(df, All_TCGA_CIN_measures %>% select(sample_id, wGII), by.x = "bcr_patient_barcode", by.y="sample_id")

df4 = df3 %>% select(bcr_patient_barcode, metastatic_one_two_three, age_at_initial_pathologic_diagnosis, type, gender, wGII)

sum(is.na(df4))

df4 %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  group_by(type) %>% 
  summarise(n = n(), n_met = sum(metastatic_one_two_three == 1),
            percent = n_met/n) %>% 
  arrange( desc(n_met))

allowed_c_types = df %>% group_by(type) %>% summarise(n = n()) %>% filter(n > 300)
df2 = df %>% filter(type %in% allowed_c_types$type)
write_csv(df4, "~/Desktop/clinical_all_cancer_with_300_samples.csv")
