library(tidyverse)
library(ggpubr)
library(readr)

all_genes_ordered_by_chr <- read_csv("pytorch_docker/TCGA_GenomeImage/data/raw_data/all_genes_ordered_by_chr.csv")
all_genes_ordered_by_chr = all_genes_ordered_by_chr[-str_detect(all_genes_ordered_by_chr$name2, "AS")]
all_genes_ordered_by_chr = all_genes_ordered_by_chr %>% filter(!chr %in% c(23,24))

write_csv(all_genes_ordered_by_chr, "pytorch_docker/TCGA_GenomeImage/data/raw_data/all_genes_ordered_by_chr_no_sex_chr.csv")

ascat <- readRDS("~/pytorch_docker/TCGA_GenomeImage/data/raw_data/filteredAscatWithRaw.rds")
ascat$log_r = log2((ascat$nAraw + ascat$nBraw)/(ascat$Ploidy))
ascat = ascat %>% filter(!Chr %in% c(23,24))
hist(ascat$log_r, breaks = 200)
ascat[which(ascat$log_r == -Inf), "log_r"] <- -5
#quantile(ascat$log_r, c(0.1, 0.9))
ascat$loss = ascat$log_r <= log2(0.5/2)
ascat$gain = ascat$log_r >= log2(4/2)
ascat = ascat %>% filter(loss == T | gain == T)
write_csv(ascat, "/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/ascat.csv")

encode_ordinal <- function(x, order = unique(x)) {
  x <- as.numeric(factor(x, levels = order, exclude = NULL))
  x
}

muts = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/muts.csv", sep=",")
muts = muts %>% group_by(sampleID) %>%
  summarize(n = n(),
            tp53 = ifelse(sum(Hugo_Symbol == "TP53") >= 1, 1, 0),
            mean_poly = mean(PolyPhen_num))

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
#c = df %>% filter(is.na(final_stage))
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Locoregional Recurrence"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Locoregional Recurrence"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Metastatic"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis|Regional lymph node"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|New Primary Tumor"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Regional lymph node"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|Distant Metastasis|Regional lymph node"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Regional lymph node|Distant Metastasis|Distant Metastasis|Distant Metastasis|Distant Metastasis|Locoregional Recurrence|Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Locoregional Recurrence|Locoregional Recurrence|Locoregional Recurrence|Distant Metastasis|Distant Metastasis"), "final_stage"] <- "Stage IV"
df[which(is.na(df$final_stage) & df$new_tumor_event_type == "Distant Metastasis|[Not Available]"), "final_stage"] <- "Stage IV"
unique(df$treatment_outcome_first_course)

a = df %>% select(clinical_stage, ajcc_pathologic_tumor_stage,stage,stage2, final_stage)
df = df %>% filter(!is.na(final_stage))
df$metastatic_one_two_three = ifelse(df$final_stage %in% c("Stage IV"), 1, 0)


ggplot(a, aes(x = type, y = percent)) + 
  geom_col() + xlab("Cancer Types") + ylab("Percent Metastatic")+
  geom_hline(yintercept = 0.53) +
  theme_pubclean()

table(df$metastatic_one_two_three)

df3 = merge(df, All_TCGA_CIN_measures %>% select(sample_id, wGII), by.x = "bcr_patient_barcode", by.y="sample_id", all.x = T)
df3 = merge(df3, muts %>% select(sampleID, tp53), by.x = "bcr_patient_barcode", by.y="sampleID", all.x = T)

df3 = df3 %>% select(bcr_patient_barcode, metastatic_one_two_three, age_at_initial_pathologic_diagnosis, type, gender, wGII,tp53, final_stage)
a = df3 %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  dplyr::group_by(type) %>% 
  dplyr::summarise(n = n(), 
                   n_1 = sum(final_stage == "Stage I"),
                   n_2 = sum(final_stage == "Stage II"),
                   n_3 = sum(final_stage == "Stage III"),
                   n_4 = sum(final_stage == "Stage IV"),
                   percent = sum(metastatic_one_two_three == 1)/n) %>%
  arrange(desc(percent)) %>%
  filter(n > 400, n_4 > 50)
a
table(df$metastatic_one_two_three)
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
quantile(a$percent, c(0.14, 0.85))
allowed_c_types = df3 %>% 
  filter(!is.na(metastatic_one_two_three)) %>%
  group_by(type) %>% 
  summarise(n = n(), 
            n_1 = sum(final_stage == "Stage I")/n,
            n_2 = sum(final_stage == "Stage II")/n,
            n_3 = sum(final_stage == "Stage III")/n,
            n_4 = sum(final_stage == "Stage IV"),
            n_met = sum(metastatic_one_two_three == 1),
            percent = n_met/n) %>% 
  filter(n > 400, n_4 > 50) %>%
  #filter(between(n_1, 0.1, 0.5)) %>% # quantile(a$percent, c(0.1, 0.9))
  arrange( desc(n_met)) 
allowed_c_types
df4 = df3 %>% filter(type %in% allowed_c_types$type)
table(df4$metastatic_one_two_three)
colnames(df4)[1] = "id"
colnames(df4)[3] = "age"
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
df$PFI.time = as.numeric(as.character(df$PFI.time))
df2 = df %>% 
  filter(!is.na(PFI), PFI != "N/A",
         PFI.time < (365*3))

allowed_c_types = df2 %>% 
  filter(!is.na(PFI)) %>%
  group_by(type) %>% 
  summarise(n = n(), 
            n_pos = sum(PFI == 1),
            n_neg = sum(PFI == 0),
            percent = n_pos/n) 
summary(allowed_c_types$n_pos)

allowed_c_types = df2 %>% 
  filter(!is.na(PFI)) %>%
  group_by(type) %>% 
  summarise(n = n(), 
            n_pos = sum(PFI == 1),
            n_neg = sum(PFI == 0),
            percent = n_pos/n) %>% 
  filter(n > 300, n_pos >= 107) %>%
  #filter(between(n_1, 0.1, 0.5)) %>% # quantile(a$percent, c(0.1, 0.9))
  arrange( desc(n_pos)) 
allowed_c_types
p1<- ggplot(allowed_c_types, aes(x = reorder(type, -percent), y  = percent)) +
  geom_col()
p2<-ggplot(allowed_c_types, aes(x = reorder(type, -n), y  = n)) +
  geom_col()
ggarrange(p1,p2)

df4 = df2 %>% filter(type %in% allowed_c_types$type) %>%
  select(bcr_patient_barcode, PFI, age_at_initial_pathologic_diagnosis, type, gender, wGII)
table(df4$PFI)
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


# PFI
df = read.delim("/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/TCGA_survival_data_clean.txt")

df2 = df %>% filter(!is.na(PFI), PFI != "N/A")
allowed_c_types = df2 %>% 
  filter(!is.na(PFI)) %>%
  group_by(type) %>% 
  summarise(n = n(), 
            n_pos = sum(PFI == 1),
            n_neg = sum(PFI == 0),
            percent = n_pos/n) 
summary(allowed_c_types$n_pos)

allowed_c_types = df2 %>% 
  filter(!is.na(PFI)) %>%
  group_by(type) %>% 
  summarise(n = n(), 
            n_pos = sum(PFI == 1),
            n_neg = sum(PFI == 0),
            percent = n_pos/n) %>% 
  filter(n > 400, n_pos >= 124) %>%
  #filter(between(n_1, 0.1, 0.5)) %>% # quantile(a$percent, c(0.1, 0.9))
  arrange( desc(n_pos)) 
allowed_c_types
df3 = df2 %>% filter(type %in% allowed_c_types$type)
table(df3$PFI)
table(df4$met)
colnames(df4)[1] = "id"
colnames(df4)[3] = "age"
df4$met = as.integer(df4$met)
write_csv(df4, "/home/mateo/pytorch_docker/TCGA_GenomeImage/data/raw_data/PFI_metadata.csv")