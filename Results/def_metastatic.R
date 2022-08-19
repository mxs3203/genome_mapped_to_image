library(tidyverse)
library(ggpubr)
encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

### Metasatatic based on stage 
df = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/TCGA_survival_data_clean.txt")

All_TCGA_CIN_measures = readRDS("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/All_TCGA_CIN_measures.rds")


unique(df$clinical_stage )
unique(df$ajcc_pathologic_tumor_stage )
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
  df$clinical_stage == "IIc" ~ "Stage II",
  df$clinical_stage == "[Discrepancy]" ~ "",
  df$clinical_stage == "[Not Applicable]" ~ "",
  df$clinical_stage == "[Unknown]" ~ "",
  df$clinical_stage == "[Discrepancy] " ~ ""
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
  df$ajcc_pathologic_tumor_stage == "IIc" ~ "Stage II",
  df$ajcc_pathologic_tumor_stage == "[Discrepancy]" ~ "",
  df$ajcc_pathologic_tumor_stage == "[Not Applicable]" ~ "",
  df$ajcc_pathologic_tumor_stage == "[Unknown]" ~ "",
  df$ajcc_pathologic_tumor_stage == "[Discrepancy] " ~ ""
)
df[which(df$stage == ""),"stage"] <- NA
df[which(df$stage2 == ""),"stage2"] <- NA
df$final_stage = unlist(apply(df, 1, function(x){
  if (!is.na(x["stage"])){
    x["stage"]
  } else if (!is.na(x["stage2"])){
    x["stage2"]
  } else {
    NA
  }
}))
c = df %>% filter(is.na(final_stage))
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis"), "final_stage"] <- "Stage IV"


a = df %>% select(clinical_stage, ajcc_pathologic_tumor_stage,stage,stage2, final_stage)
df = df %>% filter(!is.na(final_stage))
df$metastatic_one_two_three = ifelse(df$final_stage %in% c("Stage IV", "Stage III"), 1, 0)
df %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  dplyr::group_by(type) %>% 
  dplyr::summarise(n = n(), 
                   n_met = sum(metastatic_one_two_three == 1),
                   percent = n_met/n) %>%
  arrange(desc(percent))
table(df$metastatic_one_two_three)

df3 = merge(df, All_TCGA_CIN_measures %>% select(sample_id, wGII), by.x = "bcr_patient_barcode", by.y="sample_id", all.x = T)

df3 = df3 %>% select(bcr_patient_barcode, metastatic_one_two_three, age_at_initial_pathologic_diagnosis, type, gender, wGII)
a = df3 %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  dplyr::group_by(type) %>% 
  dplyr::summarise(n = n(), 
                   n_met = sum(metastatic_one_two_three == 1),
                   percent = n_met/n) %>%
  arrange(desc(percent)) %>%
  filter(n > 400)
a
p1<- ggplot(a, aes(x = reorder(type, -percent), y  = percent)) +
  geom_col() + 
  theme_minimal() +
  geom_hline(yintercept = 0.165, color = "red", linetype="dashed") + 
  geom_hline(yintercept = 0.68, color = "red", linetype="dashed") +
  theme(axis.text.x = element_text(angle = 90))
p2<-ggplot(a, aes(x = reorder(type, -n), y  = n)) +
  geom_col() + 
  theme_minimal() +
  geom_hline(yintercept = 300, color = "red",linetype="dashed") + 
  theme(axis.text.x = element_text(angle = 90))
ggarrange(p1,p2)
quantile(a$percent, c(0.15, 0.85))
allowed_c_types = df3 %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  group_by(type) %>% 
  summarise(n = n(), n_met = sum(metastatic_one_two_three == 1),
            percent = n_met/n) %>% 
  filter(n > 400) %>%
  #filter(between(percent, 0.21, 0.70)) %>% # quantile(a$percent, c(0.1, 0.9))
  arrange( desc(n_met)) 
allowed_c_types
df4 = df3 %>% filter(type %in% allowed_c_types$type)
table(df4$metastatic_one_two_three)
colnames(df4)[1] = "id"
colnames(df4)[3] = "age"
df4$type_enc = encode_ordinal(df4$type)
write_csv(df4, "/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/corrected_metastatic_based_on_stages.csv")



# PFI Version
muts = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/muts.csv", sep=",")
muts = muts %>% group_by(sampleID) %>%
   summarize(n = n(),
             tp53 = ifelse(sum(Hugo_Symbol == "TP53") >= 1, 1, 0),
             mean_poly = mean(PolyPhen_num))
df = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/TCGA_survival_data_clean.txt")
df = merge(df, All_TCGA_CIN_measures %>% select(sample_id, wGII), by.x = "bcr_patient_barcode", by.y="sample_id", all.x = T)
df = merge(df, muts %>% select(sampleID, tp53), by.x = "bcr_patient_barcode", by.y="sampleID", all.x = T)

a = df %>% 
  filter(!is.na(PFI)) %>%
  dplyr::group_by(type) %>% 
  dplyr::summarise(n = n(), 
                   n_met = sum(PFI == 1),
                   percent = n_met/n) %>%
  filter(n > 400) %>% 
  arrange(desc(percent))
a
p1<- ggplot(a, aes(x = reorder(type, -percent), y  = percent)) +
  geom_col()
p2<-ggplot(a, aes(x = reorder(type, -n), y  = n)) +
  geom_col()
ggarrange(p1,p2)

quantile(a$percent, c(0.1, 0.9))
# take top 10 cancer types with at least 300 samples
allowed_c_types = a %>% filter(between(percent, 0.153, 0.68)) %>% select(type) 
df4 = df %>% filter(type %in% allowed_c_types$type) %>%
  filter(PFI != "N/A") %>%
  select(bcr_patient_barcode, PFI, age_at_initial_pathologic_diagnosis, type, gender, wGII)
table(df4$PFI)
table(df4$tp53)
colnames(df4)[1] = "id"
colnames(df4)[3] = "age"

write_csv(df4, "/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/PFI_metadata.csv")


# Using description 
df = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/TCGA_survival_data_clean.txt")
unique(df$new_tumor_event_type)

df$met = str_detect(as.character(df$new_tumor_event_type), "Meta")


allowed_c_types = df %>% 
  filter(!is.na(met)) %>%
  dplyr::group_by(type) %>% 
  dplyr::summarise(n = n(), 
                   n_met = sum(met == 1),
                   percent = n_met/n) %>%
  filter(n > 400) %>%
  arrange(desc(percent)) %>%
  filter(between(percent, 0.1, 0.5))
df4 = df %>% filter(type %in% allowed_c_types$type) %>%
  filter(!is.na(met)) %>%
  select(bcr_patient_barcode, met, age_at_initial_pathologic_diagnosis, type, gender)
table(df4$met)
colnames(df4)[1] = "id"
colnames(df4)[3] = "age"
df4$met = as.integer(df4$met)
write_csv(df4, "/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/Met_based_on_desc_metadata.csv")


