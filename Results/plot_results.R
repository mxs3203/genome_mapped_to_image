library(tidyverse)
library(pheatmap)
library(ggrepel)
library(ggbeeswarm)
library(ggpubr)
library(patchwork)
library(ggplotify)
map2entrez <- function(gene_symbols){
  require(org.Hs.eg.db)
  entrez.alias <- as.list(org.Hs.egALIAS2EG)
  entrez.symbol <- as.list(org.Hs.egSYMBOL2EG)

  out = data.frame()
  for (g in gene_symbols) {
    enterez = entrez.symbol[g][[1]]
    if (!is.null(enterez)){
      out <- rbind(out, data.frame(gene=g, enterez=enterez ))
    } else {
      out <- rbind(out, data.frame(gene=g, enterez=g ))
    }
    
  }
  return(out)
}
newCols <- c("#A46750" ,"#A1A9AD" ,"#229487", "#EDB732" ,"#E57439" ,"#5387DD")
mycolors <- list(CancerType = newCols)

#setwd("/media/mateo/data1/genome_mapped_to_image")
setwd("~/PycharmProjects/Genome=Image")
source("~/GenomeDK_local/CancerEvolution/phd/Analysis/cnsignatures/functions.R")
vega = read.delim2("sanchez-vega-pws_1026.csv", sep = ";")
range01 <- function(x){(x-min(x))/(max(x)-min(x))}

#cancer_types =c('LUSC','LUAD', 'UCEC', 'THCA', 'COAD', 'SKCM', 'BLCA', 'KIRC', 'STAD', 'BRCA', 'OV', 'HNSC')
cancer_types =c('UCEC', 'COAD', 'BLCA', 'KIRC', 'STAD', 'OV')
scenarios = c("Metastatic", "TP53", "wGII","Age","CancerType")
data_sources = c("Methylation","Expression","Gain","Loss","Mutation")
top_n = 1000
#saveRDS(all_results, "all_results_merged_ready_for_analysis.rds")

#all_results = readRDS("all_results_merged_ready_for_analysis.rds")

############## Mandatory reading of all data


all_results = data.frame()

for (s in scenarios) {
  if (s == "TP53"){
    num_res = 38659
  } else {
    num_res = 38660
  }
  data_exp = data.frame()
  data_meth = data.frame()
  data_mut = data.frame()
  data_loss = data.frame()
  data_gain = data.frame()
  for ( type in cancer_types){
    tmp_data = read.csv(paste0("V1/",s,"/Squere/",type,"_SquereImg_",s,"_top_",num_res,".csv"))
    colnames(tmp_data)= c("rowname","attribution", "type","gene")
    tmp_data$c_type = type
    tmp_data$top_n = NA
    tmp_data$output_variable = s
    
    t = tmp_data %>% dplyr::filter(type == "Expression") %>% dplyr::arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_exp = rbind(data_exp, t)
    
    t = tmp_data %>% filter(type == "Gain") %>% dplyr::arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_gain = rbind(data_gain, t)
    
    t = tmp_data %>% filter(type == "Loss") %>% dplyr::arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_loss = rbind(data_loss, t)
    
    t = tmp_data %>% filter(type == "Mutation") %>% dplyr::arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_mut = rbind(data_mut, t)
    
    t = tmp_data %>% filter(type == "Methylation") %>% dplyr::arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_meth = rbind(data_meth, t)
  }
  
  data_meth = data_meth
  data_exp = data_exp 
  data_mut = data_mut 
  data_gain = data_gain 
  data_loss = data_loss 
  total_data = rbind(data_meth, data_mut, data_exp, data_gain, data_loss)
  
  all_results = rbind(all_results, total_data)
}
all_results$gene = as.character(all_results$gene)
all_results[which(all_results$gene == "SMURF2P1-LRRC37BP1"),"gene"] <- "SMURF2P1"

tmp_tmp = data.frame()

for (d_s in cancer_types){
  tmp = all_results %>% filter(c_type == d_s)
  tmp = tmp %>% mutate(scaled_attribtuion = range01(attribution)) 
  tmp_tmp = rbind(tmp_tmp, tmp)
}
nrow(all_results) == nrow(tmp_tmp)
all_results = tmp_tmp
############## END OF Mandatory reading of all data

mean(all_results$scaled_attribtuion)
############### Figure 3 summarizing sources and cancer types
library(wesanderson)

summ_res = all_results %>% 
  dplyr::filter(top_n, output_variable=="Metastatic") %>%
  dplyr::group_by(type, c_type,output_variable) %>%
  dplyr::summarize(m = mean(attribution)) 

# Supplementary Figure 7 and Figure 3A
ggplot(summ_res, aes(x =output_variable , y = m, fill = type)) + 
  geom_col(position="dodge")+
  ylab("Attribution")+
  scale_fill_manual(values=wesanderson::wes_palette(n=5, name="Darjeeling2"))+
  theme_pubclean()+ 
  theme(axis.text.x = element_text(angle=45))

# Supplementary Figure 7
summ_res = all_results %>% 
  dplyr::filter(top_n) %>%
  dplyr::group_by(type, c_type,output_variable) %>%
  dplyr::summarize(m = mean(attribution)) 

ggplot(summ_res, aes(x =reorder(output_variable, -m), y = m, fill = type)) + 
  geom_col(position="dodge")+
  ylab("Attribution")+
  scale_fill_manual(values=wesanderson::wes_palette(n=5, name="Darjeeling2"))+
  theme_pubclean()+ 
  facet_wrap(~c_type)

m = matrix( nrow = 5, ncol = 6)
cnt = 1
for (i in 1:5){
  for (j in 1:6){
    m[i,j] = summ_res[cnt, "m"][[1]]
    cnt = cnt + 1
  }
}
pheatmap(m)


############### ############## ##############  Chromo map
all_genes = read.delim("../data/raw_data/all_genes.csv", sep = ",")

library(chromoMap)
chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "cenntstart")


ann_total = data.frame()
for (c in cancer_types){
  ann = data.frame()
  genes = (all_results %>% 
                   dplyr::filter(c_type == c, output_variable == "Metastatic", (type == "Methylation")) %>%
                   dplyr::arrange(desc(scaled_attribtuion)) %>% 
                   head(100) %>% 
                   dplyr::select(gene, scaled_attribtuion) )
  
  for (i in 1:nrow(genes)){
    g = genes[i, "gene"]
    val = genes[i, "scaled_attribtuion"]
    ind = which(all_genes[,"name2"]  == g)
    ann<- rbind(ann, data.frame(V1 = all_genes[ind, "name2"], V2=all_genes[ind, "chr"],V3=all_genes[ind, "start"], V4=all_genes[ind, "end"]) )
  }
  ann$cancerType = c
  ann_total <- rbind(ann_total, ann)
}

chrom_to_viz = 11
chromoMap(list(chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart"), 
               chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart") ,
               chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart") ,
               chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart"),
               chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart") ,
               chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart")), 
          list(ann_total %>% filter(cancerType == "UCEC"), ann_total %>% filter(cancerType == "STAD"),
               ann_total %>% filter(cancerType == "OV"), ann_total %>% filter(cancerType == "KIRC"),
               ann_total %>% filter(cancerType == "COAD"), ann_total %>% filter(cancerType == "BLCA")),
          ploidy = 6,
          n_win.factor = 1, 
          segment_annotation = T,
          data_based_color_map = T,
          data_type = c( "categorical"),
          data_colors = list(mycolors$CancerType),
          legend = T,
          #labels = T,
          chr_curve = 8,
          interactivity = F,
          lg_x = 150,
          lg_y= 500,
          export.options = F,
            
          )

chromoMap(list(chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart")), 
          list(ann_total),
          n_win.factor = 1, 
          segment_annotation = T,
          data_based_color_map = T,
          data_type = c( "categorical"),
          data_colors = list(mycolors$CancerType),
          legend = T,
          #labels = T,
          chr_curve = 8,
          interactivity = F,
          lg_x = 150,
          lg_y= 500,
          export.options = F,
          
)



table(ann_total$cancerType)




chromoMap(list(chrominfo_hg19 %>% select("chrom", "chrstart", "chrlength", "centstart")),list(ann))
  
############## Reactome Pathways
quantile(all_results$scaled_attribtuion, probs = c(0.10, 0.90, 0.95, 0.99))
  
library(ReactomePA)
reactome_res = data.frame()
for (c in cancer_types){
  for (t in data_sources) {
    print(paste0(c, t))
    tmp = all_results %>% dplyr::filter(output_variable == "Metastatic", 
                                        c_type == c,
                                        type == t) %>%
      dplyr::arrange(desc(scaled_attribtuion)) %>% 
      head(100) #%>% 
      #filter(scaled_attribtuion >=  0.07278) # 
    if(nrow(tmp) > 0){
      mapping = map2entrez(as.character(tmp$gene))
      tmp = merge(tmp, mapping, by = "gene", all.x= T)
      x <- enrichPathway(gene=tmp$enterez, 
                         pvalueCutoff=0.01,
                         pAdjustMethod = "bonferroni",
                         readable=T)
      
      if(!is.null(x)){
        x = x@result
        x = x %>% filter(p.adjust < 0.05) %>%
          arrange(p.adjust) %>%
          head(10)
        if(nrow(x) > 0){
          x$c_type = c
          x$type = t
          reactome_res <- rbind(reactome_res, data.frame(x))
        }
      }
     
    }
  }
  #break
}  
reactome_res$geneRatio = unlist(lapply(reactome_res$GeneRatio, function(x){
  nums = str_split(x, "/")
  as.numeric(nums[[1]][1])/as.numeric(nums[[1]][2])
}))
ggplot(reactome_res , aes(x = Description, 
                         y = geneRatio, 
                         size = p.adjust,
                         color = type)) + 
  geom_point() +
  theme_minimal() +
  theme(text = element_text(size = 9)) +
  scale_size(trans = 'reverse') +
  coord_flip()+
  theme(axis.text.x = element_text(angle = 90)) +
  facet_wrap(~c_type)

############## End of Reactome Pathways




############## # Figure 2 Encoded genomes UMAP #########
library(umap)

pc = umap(encoded_genomes[, 2:513],  min_dist=1, spread = 2, n_neighbours = 5)
res = as.data.frame(pc$layout)
res$met = meta_data$metastatic_one_two_three
ggplot(res, aes(x =V1, y = V2, color = met)) +
  geom_point()

color_for_plt = newCols[1:(length(unique(res$var)))]
names(color_for_plt) = NULL
metadata = read.csv("main_meta_data.csv")
for (param in c(scenarios, "RandomCancerType")){
  #param = "Metastatic"
  tmp_data = read.csv(paste0("ResultsV1/",param,"/encoded_genomes.csv"))
  var = case_when(
    param == "TP53" ~ "tp53",
    param  == "wGII" ~ "wGII",
    param  == "Metastatic" ~ "met",
    param  == "Age" ~ "age_at_initial_pathologic_diagnosis",
    param  == "CancerType" ~ "type",
    param  == "RandomCancerType" ~ "type",
    TRUE ~ as.character(param)
  )
  tmp_data = merge(metadata %>% 
                     #dplyr::filter(type == "KIRC") %>%
                     dplyr::select("id", var) , tmp_data, by.x="id", by.y = "sampleid")
  umap_res = umap(tmp_data[, paste0("X", 0:127)], min_dist=1, spread = 5)# 1,5
  res = data.frame(umap_res$layout)
  res$sampleid = tmp_data[,"id"]
  res$var = tmp_data[,var]
  
  if (var %in% c("tp53", "met", "type")){
    res$var = as.factor(res$var)
    if (var %in% c("tp53", "met")){
      assign(x= paste0("p_",param), value = ggplot(res, aes(x = X1, y = X2, color = var)) +
               geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
               ggtitle(param) +
               scale_color_manual(values=color_for_plt[2:3]) +
               theme_pubclean())
    } else { # Type and Rand Type
        if (param == "CancerType"){ # type
          assign(x= paste0("p_",param), value = ggplot(res, aes(x = X1, y = X2, color = var)) +
                 geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
                 ggtitle(param) +
                  scale_color_manual(values = c("BLCA"=newCols[6][[1]], "OV"=newCols[3][[1]], 
                                                "COAD"=newCols[5][[1]], "STAD"=newCols[2][[1]],
                                                "KIRC"=newCols[4][[1]], "UCEC"=newCols[1][[1]])) +
                 theme_pubclean()) 
          } else { # rand Type
          res$var = sample(c("BLCA", "OV", "KIRC","STAD", "UCEC","COAD"),replace = T, size = nrow(res))
          assign(x= paste0("p_",param), value = ggplot(res, aes(x = X1, y = X2, color = var)) +
               geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
               ggtitle(param) +
                scale_color_manual(values = c("BLCA"=newCols[6][[1]], "OV"=newCols[3][[1]], 
                                              "COAD"=newCols[5][[1]], "STAD"=newCols[2][[1]],
                                              "KIRC"=newCols[4][[1]], "UCEC"=newCols[1][[1]])) + 
               theme_pubclean())
          }
    }
  } else { # wGII, Age
    assign(x= paste0("p_",param), value = ggplot(res, aes(x = X1, y = X2, color = var)) +
             geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
             ggtitle(param) +
             theme_pubclean())
  }
 
}

cowplot::plot_grid(p_wGII, p_TP53, p_Metastatic, p_CancerType,p_RandomCancerType, p_Age)

# Clustering of UMAP and KM Curves
fviz_nbclust(res[,2:3], pam(), method = "silhouette", k.max = 10) + theme_minimal() + ggtitle("The Silhouette Plot")

gap_stat <- clusGap(res[,2:3], FUN = pam, nstart = 50, K.max = 10, B = 20)
fviz_gap_stat(gap_stat) + theme_minimal() + ggtitle("fviz_gap_stat: Gap Statistic")

cl = pam(res[,c("X1","X2")], k = 3)
res$kmeans_cluster = as.factor(cl$clustering)
ggplot(res, aes(x = X1, y = X2, color = kmeans_cluster)) +
  geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
  ggtitle(param) +
  theme_pubclean() 

res = merge(res, clinical %>% dplyr::select("bcr_patient_barcode", "DSS","DSS.time"), by.x="sampleid",by.y = "bcr_patient_barcode")
surv_os <- Surv(as.numeric(as.character(res[,'DSS.time']))/365, as.numeric(as.character(res[,'DSS'])))
fit_os <- survfit(survplot::censor(surv_os, 5)~kmeans_cluster, data = res)
makeSurvPlot(fit_os, 
             '', 
             legen_title = "", 
             ylab = "OS",
             legend_coord = c(0.1, 0.3),
             colors = c(MY_COLORS_red_blue))


############## END OF Encoded genomes UMAP #########

library(RColorBrewer)


# Sanchez Vega Grouping Figure 3 Panel A

#all_results$attribution
heatmap_res = data.frame()
types = c()
for (c in unique(all_results$c_type)){
  types = c(types, rep(c, 10))
  tmp = all_results %>% filter(c_type == c, 
                               output_variable == "Metastatic",
                               )
  
  results = data.frame()
  for (i in 1:length(unique(vega$Pathway_pretty))) {
    vega_group = vega %>% filter(Pathway_pretty == unique(vega$Pathway_pretty)[i]) 
    for (j in 1:length(unique(tmp$type))){
      dlbc_tmp = tmp %>% filter(type == unique(tmp$type)[j])
      num = sum(dlbc_tmp[which(dlbc_tmp$gene %in% vega_group$Gene), "scaled_attribtuion"]) #attribution, scaled_attribtuion
      results[i,j] = num
      all_results$attribution
    } 
  }
  colnames(results) = unique(tmp$type)
  rownames(results) = paste0(unique(vega$Pathway_pretty),"_", c)
  heatmap_res <- rbind(heatmap_res, results)
}
ann_df = data.frame(CancerType=types)
rownames(ann_df) = rownames(heatmap_res)
heatmap_res = t(heatmap_res)

newCols <- c("#A46750" ,"#A1A9AD" ,"#229487", "#EDB732" ,"#E57439" ,"#5387DD")
names(newCols) <- unique(ann_df$CancerType)
mycolors <- list(CancerType = newCols)


names(which(colMeans(heatmap_res) > mean(heatmap_res)))
heatmap_res_ = heatmap_res[,names(which(colMeans(heatmap_res) > mean(heatmap_res)))]
pheatmap(heatmap_res_,
         cluster_rows = F,
         cluster_cols = T,
         scale="row",
         cutree_cols = 7,
         clustering_method = "ward.D",
         annotation_col = ann_df,
        cellwidth = 10,
        annotation_colors = mycolors,
        cellheight = 10)

# Figure 3 Panel B - Individual genes
cosmic_genes = read.delim ("COSMIC_cancerGeneCensus_20220307_14_48_24 2022.tsv")
cosmic_genes_oncogenes = cosmic_genes 
known_genes = as.character(cosmic_genes$Gene.Symbol)

genes = data.frame()
for (c in unique(all_results$c_type)) {
  ttmp = all_results %>% 
    filter(output_variable == "Metastatic",
           c_type == c) %>% 
    dplyr::arrange(desc(scaled_attribtuion)) %>% 
    head(40) # 40 is the best
  genes <- rbind(genes, ttmp)
}

genes[order(reorder(genes$gene,-genes$scaled_attribtuion)), "gene"]  %in% known_genes
sum(genes[order(reorder(genes$gene,-genes$scaled_attribtuion)), "gene"]  %in% known_genes)
list_of_genes_plot = genes[order(reorder(genes$gene,-genes$scaled_attribtuion)), "gene"]
length(unique(list_of_genes_plot))
gene_name_color = ifelse( unique(list_of_genes_plot) %in% known_genes,  "#003DAA", "#CC3D3D")
cat("Genes that should be blue",unique(list_of_genes_plot[list_of_genes_plot %in% known_genes]), "\n")

ggplot(genes, aes(x = reorder(gene,-scaled_attribtuion), y = scaled_attribtuion, color = c_type, shape = type )) +
  geom_point() + 
  ylab("Attribution - Scaled by Cancer Type") + xlab("Genes") + 
  theme_light() +
  scale_shape_manual(values=c(16, 1, 17,15))+
  scale_color_manual(values = c("BLCA"=newCols[6], "OV"=newCols[3], 
                                "COAD"=newCols[5], "STAD"=newCols[2],
                                "KIRC"=newCols[4], "UCEC"=newCols[1]))+
  theme(axis.text.x = element_text(angle = 90, color = gene_name_color, hjust = 0.5, vjust = 0.5))


# Figure 4: KM Curves and coxph
setwd("~/PycharmProjects/Genome=Image")
gene_exp_tcga <- readRDS("/Users/au589901/GenomeDK_local/CancerEvolution/phd/Analysis/Sting-cGAS_Paper/All_TCGA_GENE_EXP.rds")
suppressMessages(library(survminer))
suppressMessages(library(survplot))
source("../../GenomeDK_local/CancerEvolution/phd/Analysis/cnsignatures/functions.R")
clinical <- read.delim2("/Users/au589901/GenomeDK_local/CancerEvolution/phd/Datasets/TCGA/TCGA_survival_data_clean.txt", sep = '\t') 
clinical$cancer_stage = as.factor(case_when(
  clinical$ajcc_pathologic_tumor_stage  == "Stage IV" ~ "IV",
  clinical$clinical_stage  == "Stage IVB" ~ "IV",
  clinical$clinical_stage  == "Stage IVA" ~ "IV",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIIA"  ~ "III",
  clinical$ajcc_pathologic_tumor_stage  == "Stage IB" ~ "I",
  clinical$ajcc_pathologic_tumor_stage == "Stage IA" ~ "I",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIIB" ~ "III",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIB" ~ "II",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIA" ~ "II",
  clinical$ajcc_pathologic_tumor_stage == "Stage III" ~ "III",
  clinical$ajcc_pathologic_tumor_stage == "Stage II" ~ "II",
  clinical$ajcc_pathologic_tumor_stage == "Stage I" ~ "I",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIC" ~ "II",
  clinical$ajcc_pathologic_tumor_stage == "Stage IIIC" ~ "III",
  clinical$ajcc_pathologic_tumor_stage == "I/II NOS" ~ "II",
  clinical$ajcc_pathologic_tumor_stage == "Stage 0" ~ "0",
  clinical$ajcc_pathologic_tumor_stage == "IS" ~ "I",
  clinical$ajcc_pathologic_tumor_stage == "Stage X" ~ "III",
  clinical$ajcc_pathologic_tumor_stage == "1" ~ "I",
  clinical$ajcc_pathologic_tumor_stage == "0" ~ "Stage 0",
  TRUE ~ "Not Available"
))

clinical$clinical_stage = as.factor(case_when(
  clinical$clinical_stage  == "Stage IV" ~ "IV",
  clinical$clinical_stage  == "Stage IVA" ~ "IV",
  clinical$clinical_stage  == "Stage IVB" ~ "IV",
  clinical$clinical_stage == "Stage IIIA"  ~ "III",
  clinical$clinical_stage  == "Stage IB" ~ "I",
  clinical$clinical_stage == "Stage IA" ~ "I",
  clinical$clinical_stage == "Stage IIIB" ~ "III",
  clinical$clinical_stage == "Stage IIB" ~ "II",
  clinical$clinical_stage == "Stage IIA" ~ "II",
  clinical$clinical_stage == "Stage III" ~ "III",
  clinical$clinical_stage == "Stage II" ~ "II",
  clinical$clinical_stage == "Stage I" ~ "I",
  clinical$clinical_stage == "Stage IIC" ~ "II",
  clinical$clinical_stage == "Stage IIIC" ~ "III",
  clinical$clinical_stage == "I/II NOS" ~ "II",
  clinical$clinical_stage == "Stage 0" ~ "0",
  clinical$clinical_stage == "IS" ~ "I",
  clinical$clinical_stage == "Stage X" ~ "III",
  clinical$clinical_stage == "1" ~ "I",
  clinical$clinical_stage == "0" ~ "Stage 0",
  TRUE ~ "Not Available"
))

clinical[which(clinical$type == "OV"),"cancer_stage"] = clinical[which(clinical$type == "OV"),"clinical_stage"]
#clinical[which(clinical$type == "UCEC"),"cancer_stage"] = clinical[which(clinical$type == "UCEC"),"clinical_stage"]

t = clinical %>% filter(type =="COAD") 
table(t$PFI)

clinical$metastatic_one_two_three = ifelse(clinical$cancer_stage %in% c("0","I","II", "III"), 1, 0)

clinical %>% 
  filter(type == "UCEC") %>%
  summarise(n= n(),
            nonmets = sum(metastatic_one_two_three == 1)+1,
            mets= sum(metastatic_one_two_three == 0)+1)

clinical %>% 
  filter(type == "UCEC") %>% 
  summarise(n = n(), sum(new_tumor_event_site_other  != "N/A"))

clinical[which(clinical$type == "UCEC" & clinical$new_tumor_event_site_other != "N/A"), "metastatic_one_two_three"] <- 1

prop = clinical %>% group_by(type) %>%
  summarise(n = n(),
            mets = sum(metastatic_one_two_three == 1)+1,
            nonmets = sum(metastatic_one_two_three == 0)+1,
            prop = mets/n) %>%
  filter(n > 400, (mets >= 1 & nonmets >= 1), prop <= 0.85, prop > 0.15) %>%
  arrange(desc(prop))

library(ggmosaic)


ggplot(prop, aes( x = reorder(type, -prop), y = prop)) +
  geom_col() +
  theme_minimal() + 
  xlab("Cancer types") + 
  ylab("Proportion of metastatic samples")


clinical <- clinical %>% dplyr::filter(type %in% c("BLCA", "COAD","KIRC", "OV","STAD", "UCEC"))
gene_exp_tcga_blca <- gene_exp_tcga[,which(colnames(gene_exp_tcga) %in% clinical$bcr_patient_barcode)]
TIL <- calculateTIL(gene_exp_tcga_blca,immune_genes = immune_genes, dataType = "lCPM")
TIL$sample_id <- rownames(TIL)
mutation <- readRDS("ALL_TCGA_MAF_summary.rds")
CIN <- readRDS("/Users/au589901/GenomeDK_local/CancerEvolution/phd/Datasets/TCGA/All_TCGA_CIN_measures.rds")
which(sting_cgas_pathway %in% rownames(gene_exp_tcga))
tcga_mutation = readRDS("/Users/au589901/GenomeDK_local/CancerEvolution/phd/Datasets/TCGA/driver_mutations_all_tcga.rds")
gene_exp_tcga_blca <- gene_exp_tcga_blca[which(rownames(gene_exp_tcga) %in% c(sting_cgas_pathway,hk_genes)),]
gene_exp_tcga_blca <- as.data.frame(t(gene_exp_tcga_blca))
gene_exp_tcga_blca$sample_id <- rownames(gene_exp_tcga_blca)
Total_TCGA <- gene_exp_tcga_blca
Total_TCGA <- na.omit(Total_TCGA)

Total_TCGA <- merge(Total_TCGA,clinical, by.x = "sample_id", by.y = "bcr_patient_barcode",all.x = T)
Total_TCGA <- merge(Total_TCGA,CIN, by = "sample_id", all.x = T)
Total_TCGA <- merge(Total_TCGA,TIL, by = "sample_id", all.x = T)
for_conf_mat = read.delim2("~/Desktop/GMI_Paper/confusion_matrix.csv", sep = ",")
for_conf_mat = merge(for_conf_mat, clinical %>% select("bcr_patient_barcode", "type"), by.x="sampleid", by.y = "bcr_patient_barcode", all.x = T)

for_conf_mat = for_conf_mat %>%
  group_by(type) %>%
  summarise(n = n(), 
            correct = sum(true_y == pred_y), 
            prop_cor = correct/n)

ggplot(for_conf_mat, aes(x = type, y = prop_cor, fill = type)) +
  geom_col() +
  theme_pubclean()+
  scale_fill_manual(values = c("BLCA"=newCols[6][[1]], "OV"=newCols[3][[1]], 
                                "COAD"=newCols[5][[1]], "STAD"=newCols[2][[1]],
                                "KIRC"=newCols[4][[1]], "UCEC"=newCols[1][[1]])) +
  scale_y_continuous( limits=c(0, 1), breaks = c(0.5,0.6,0.7, 0.8, 0.9,1))
  
  


Total_TCGA %>% group_by(type) %>%
  summarise(n = n(),
            mets = sum(metastatic_one_two_three == 1),
            nonmets= sum(metastatic_one_two_three == 0),
            prop = mets/nonmets) %>% 
  arrange(desc(n))

MY_COLORS_red_blue = c("#4292c6","#aaaaaa", "#cb181d" ) 


unlist(all_results %>% 
                 dplyr::filter(c_type == "BLCA", output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                 dplyr::arrange(desc(scaled_attribtuion)) %>% 
                 head(10) %>% 
                 dplyr::select(gene) )

gene_exp_tcga <- readRDS("/Users/au589901/GenomeDK_local/CancerEvolution/phd/Analysis/Sting-cGAS_Paper/All_TCGA_GENE_EXP.rds")

"MIR3647" %in% rownames(gene_exp_tcga)
# genes are from Figure 3

for (c in cancer_types){
  genes = unlist(all_results %>% 
                   dplyr::filter(c_type == c, output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                         dplyr::arrange(desc(scaled_attribtuion)) %>% 
                        head(11) %>% 
                   dplyr::select(gene) )
  
  km_res_tcga = data.frame()
  cox_res_tcga = data.frame()
  wilcox_res_tcga = data.frame()
  for(g in genes){
    if (g != "FAM153C" & g != "FAM84B" & g != "SNORD111B"){
      cat(c, " - ", g, "\n")
      tmp_tcga = Total_TCGA %>% filter(type == c) 
      
      ids = which(colnames(gene_exp_tcga) %in% tmp_tcga$sample_id)
      exp = data.frame(t(gene_exp_tcga[g, ids]))
      exp$gene = exp[,1]
      exp$sample_id = rownames(exp) 
      
      tmp = merge(exp,Total_TCGA %>% dplyr::select("sample_id", "PFI", "PFI.time", "metastatic_one_two_three","cancer_stage","histological_grade","gender", "race","age_at_initial_pathologic_diagnosis" ), by = "sample_id")
      surv_os <- Surv(as.numeric(as.character(tmp[,'PFI.time']))/365, as.numeric(as.character(tmp[,'PFI'])))
      #tmp$impact = gtools::quantcut(tmp$gene, 3)
      #fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
      # p_val = surv_pvalue(fit_os)[[2]]
      # km_res_tcga <- rbind(km_res_tcga, data.frame(gene = g, pval = p_val, quantcut = 3, data = "Expression"))
      # tmp$metastatic_one_two_three = as.character(tmp$metastatic_one_two_three)
      # tmp_1 = tmp %>% dplyr::filter(metastatic_one_two_three =="1")
      # tmp_2 = tmp %>% dplyr::filter(metastatic_one_two_three == "0")
      # if(nrow(tmp_1) > 1 & nrow(tmp_2)) {
      #   t = ggpubr::compare_means(formula = gene ~ metastatic_one_two_three, data = tmp)
      #   t$gene = g
      #   t$ratio = mean(tmp_1$gene)/mean(tmp_2$gene)
      #   wilcox_res_tcga <- rbind(wilcox_res_tcga, t)
      # }
      # 
      tmp$age = scale(as.numeric(tmp$age_at_initial_pathologic_diagnosis),center = T, scale = T)
      if (c %in% c( "OV" , "UCEC")) {
        cox_model = coxph(formula = surv_os ~ gene + cancer_stage + race + age, data = tmp, robust = T, model = T)
      } else {
        cox_model = coxph(formula = surv_os ~ gene + cancer_stage + race + age, data = tmp, robust = T, model = T)
      }
      sum_fit = summary(cox_model)
      for_forest = as.data.frame(sum_fit$coefficients)[1,]
      for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
      #for_forest$param = rownames(for_forest)
      for_forest$gene = g
      cox_res_tcga <- rbind(cox_res_tcga, for_forest)
    }
    
  }
  #km_res_tcga$p_ajd = p.adjust(km_res_tcga$pval, method = "fdr")
  #wilcox_res_tcga$p_ajd = p.adjust(wilcox_res_tcga$p, method = "fdr")
  #cox_res_tcga$p_ajd = p.adjust(cox_res_tcga$`Pr(>|z|)`, method = "fdr")
  cox_res_tcga$p_ajd = cox_res_tcga$`Pr(>|z|)`
  breaks = seq(-2,2,by = 0.5)
  # and labels
  labels = as.character(breaks)
  assign(value= ggplot(data=cox_res_tcga,
                       aes(x = paste0(gene),y = `coef`, ymin =`X2.5..`, ymax = `X97.5..`))+
           geom_pointrange(aes(col=`exp(coef)`))+
           geom_hline(yintercept =0, linetype=2) + 
           geom_text(aes(label=paste0(signif.num(p_ajd)),y = 2)) + 
           geom_errorbar(aes(ymin=`X2.5..`,ymax=`X97.5..`,col=`coef`),width=0.5,cex=1)+
           theme(plot.title=element_text(size=16,face="bold"),
                 axis.text.x=element_text(face="bold"),
                 axis.title=element_text(size=12,face="bold"))+
           
           coord_flip() + 
           theme_pubclean() + 
           xlab("") + 
           ggtitle(paste0("TCGA - ",c," Expression genes: \nCoxPH Univariate Models")) + 
           scale_color_gradient(guide = F, low = "#000000", high = "#000000") +
           scale_y_continuous(limits = c(-2, 2), breaks = breaks, labels = labels,
                              name = "HR"), x = paste0(c))
    
}

ggarrange(BLCA, KIRC, UCEC, COAD, OV,STAD)


# Figure 5
data <- data.frame(
  group=LETTERS[1:5],
  value=c(13,7,9,21,2)
)
data <- data %>% 
  arrange(desc(group)) %>%
  mutate(prop = value / sum(data$value) *100) %>%
  mutate(ypos = cumsum(prop)- 0.5*prop )

# Basic piechart
ggplot(data, aes(x="", y=prop, fill=group)) +
  geom_bar(stat="identity", width=1, color="white") +
  coord_polar("y", start=0) +
  theme_void() + 
  theme(legend.position="none") +
  
  geom_text(aes(y = ypos, label = group), color = "white", size=6) +
  scale_fill_brewer(palette="Set1")

top_n = 10
par(mfrow=c(2,3) ) 
for (var in unique(all_results$c_type)){
  f5_data = all_results %>% 
    filter(output_variable == "Metastatic", c_type == var ) %>% 
    arrange(desc(scaled_attribtuion)) %>%  #attribution, scaled_attribtuion
    mutate(name = paste0(gene," ", type)) %>% 
    head(top_n) %>% 
    arrange(desc(scaled_attribtuion))
  f5_data [1:top_n, "show_name"] = f5_data [1:top_n, "name"]
  pie(f5_data$scaled_attribtuion,
      f5_data$show_name,  
      col = mycolors$CancerType,main = var)
}


top_n = 10
par(mfrow=c(6,5) ,mai=c(0.1,0.1,0.1,0.1)) 
for (var2 in unique(all_results$c_type)){
  for (var in unique(all_results$output_variable)){
    f5_data = all_results %>% 
      filter(output_variable == var, c_type == var2 ) %>% 
      arrange(desc(attribution)) %>%  #attribution, scaled_attribtuion
      mutate(name = paste0(gene," ", type)) %>% 
      head(top_n) %>% 
      arrange(desc(attribution))
    f5_data [1:top_n, "show_name"] = f5_data [1:top_n, "name"]
    pie(f5_data$attribution,
        f5_data$show_name,  
        col = mycolors$CancerType,main = var)
    }
}
 
####################################################### Validation on independet cohorts ( Figure 4 or 5) #######################################################
clinical = read.delim("~/GenomeDK_local/CancerEvolution/phd/Datasets/UROMOL_BLCA/UROMOL2_FU.txt")
uromol_exp = readRDS("~/GenomeDK_local/CancerEvolution/phd/Datasets/UROMOL_BLCA/gene_exp_standardized_logCPM_no_dups_matrix.rds")

genes_blca = unlist(all_results %>% 
  filter(c_type == "BLCA", output_variable == "Metastatic", 
         (type == "Expression" | type == "Methylation"), 
         gene != "SNORD111B") %>%
  dplyr::arrange(desc(scaled_attribtuion)) %>% 
  head(10) %>% 
   dplyr::select(gene) )

clinical$Smoking_status = as.factor(case_when(
  clinical$Smoking  == "Current" ~ "Yes",
  clinical$Smoking == "Former"  ~ "Yes",
  clinical$Smoking == "Never"  ~ "No")
)

cox_res_uromol = data.frame()
km_res_uromol = data.frame()
wilcox_uromol = data.frame()
tt = data.frame()
for (g in genes_blca){
  exp = data.frame(t(uromol_exp[g,]))
  exp$SampleID = rownames(exp) 
  exp$gene = exp[,1]
  
  tmp = merge(clinical, exp, by = "SampleID")
  surv_os <- Surv(as.numeric(as.character(tmp[,'PFS_60_months_fu']))/12, as.numeric(as.character(tmp[,'Progression_60_months_fu'])))
  # tmp$impact = gtools::quantcut(tmp$gene, 2)
  # fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
  # p_val = surv_pvalue(fit_os)[[2]]
  # km_res_uromol <- rbind(km_res_uromol, data.frame(gene = g, pval = p_val, quantcut = 2))
  # tmp$impact = gtools::quantcut(tmp$gene, 3)
  # fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
  # p_val = surv_pvalue(fit_os)[[2]]
  # km_res_uromol <- rbind(km_res_uromol, data.frame(gene = g, pval = p_val, quantcut = 2, data = "Expression"))
  # 
  cox_model = coxph(formula = surv_os ~ gene + Gender + Age + Tumor_stage , data = tmp, robust = T, model = T)
  sum_fit = summary(cox_model)
  for_forest = as.data.frame(sum_fit$coefficients)[1,]
  for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
  #for_forest$param = rownames(for_forest)
  for_forest$gene = g
  #tt <- rbind(tt, for_forest %>% dplyr::select("gene", "coef", "z", "Pr(>|z|)", "X2.5..", "X97.5.."))
  
  cox_res_uromol <-rbind(cox_res_uromol, for_forest)
  
 
  tmp_1 = tmp %>% filter(Tumor_grade == "High grade" )
  tmp_2 = tmp %>% filter(Tumor_grade == "Low grade")
  t = compare_means(formula = gene ~ Tumor_grade, data = tmp)
  t$gene = g
  t$ratio = mean(tmp_1$gene)/mean(tmp_2$gene)
  wilcox_uromol <- rbind(wilcox_uromol, t)
  
}
wilcox_uromol$adj_p = p.adjust(wilcox_uromol$p.adj, method = "fdr")
wilcox_uromol$gene = unlist(apply(wilcox_uromol,1, function(x){
  if(as.numeric(x["adj_p"]) < 0.05){
    x["gene"]
  } else {
    ""
  }
  
}))
ggplot(wilcox_uromol , aes(x = ratio, y = -log10(adj_p), label = gene))+
  geom_point() +
  geom_text_repel()+
  xlab("Mean Ratio between High Grade vs Low Grade")+ 
  ggtitle("UROMOL: High Grade vs Low Grade") +
  geom_hline(yintercept = -log10(0.05),color = "red", linetype="dashed" ) +
  geom_vline(xintercept = 1,color = "black", linetype="dashed" ) +
  theme_pubclean() + 
  theme(axis.text.x = element_text(angle = 90)) 


cox_res_uromol$adj_p = p.adjust(cox_res_uromol$`Pr(>|z|)`, method = "fdr")
breaks = seq(-2,2,by = 0.5)
# and labels
labels = as.character(breaks)
ggplot(data=cox_res_uromol,
       aes(x = paste0(gene),y = `coef`, ymin =`X2.5..`, ymax = `X97.5..`))+
  geom_pointrange(aes(col=`exp(coef)`))+
  geom_hline(yintercept =0, linetype=2) + 
  geom_text(aes(label=paste0(signif.num(`adj_p`)),y = 2)) + 
  geom_errorbar(aes(ymin=`X2.5..`,ymax=`X97.5..`,col=`coef`),width=0.5,cex=1)+
  theme(plot.title=element_text(size=16,face="bold"),
        axis.text.x=element_text(face="bold"),
        axis.title=element_text(size=12,face="bold"))+
  
  coord_flip() + 
  theme_pubclean() + 
  xlab("") + 
  ggtitle(paste0("UROMOL Top 10 Gene Expression/Methylation")) + 
  scale_color_gradient(guide = F, low = "#000000", high = "#000000") +
  scale_y_continuous(limits = c(-2, 2), breaks = breaks, labels = labels,
                     name = "HR")



# Random genes coxph
bootstrap_res = data.frame()
for (i in 1:1000){
  random_genes_res = data.frame()
  random_genes = sample(rownames(uromol_exp),size = 10)
  for (g in random_genes){
    exp = data.frame(t(uromol_exp[g,]))
    exp$SampleID = rownames(exp) 
    exp$gene = exp[,1]
    
    tmp = merge(clinical, exp, by = "SampleID")
    surv_os <- Surv(as.numeric(as.character(tmp[,'PFS_60_months_fu']))/12, as.numeric(as.character(tmp[,'Progression_60_months_fu'])))
    cox_model = coxph(formula = surv_os ~ gene +Gender + Age , data = tmp, robust = T, model = T)
    sum_fit = summary(cox_model)
    for_forest = as.data.frame(sum_fit$coefficients)[1,]
    for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
    #for_forest$param = rownames(for_forest)
    for_forest$gene = g
    random_genes_res <-rbind(random_genes_res, for_forest)
  }
  random_genes_res$adj_p = p.adjust(random_genes_res$`Pr(>|z|)`, method = "fdr")
  bootstrap_res <- rbind(bootstrap_res, data.frame(num_sign_gmi = 7,
                         proportion_gmi = 6/10, num_random_sign = sum(random_genes_res$adj_p < 0.05), 
                         proportion_random = sum(random_genes_res$adj_p < 0.05)/10))
}

median(bootstrap_res$proportion_random) # 0.2
median(bootstrap_res$proportion_gmi) #0.7

pie_colors = c("#1e8479", "#a5eae2", "#caa394","#a46750" )
pie(c(0.7,0.3,0.7,0.3),
    c("70%","30%", "80%", "20%"),  
    col = pie_colors,
    main = "")
legend(0.9, 0.2, c("Proportion of significant genes Found by GMI","Proportion of not significant genes Found by GMI",
                   "Proportion of not significant genes sampled randomly", "Proportion of significant genes sampled randomly"), cex = 0.8, fill = pie_colors)

# gains/losses
uromol_ascat = read.delim("~/GenomeDK_local/CancerEvolution/phd/Datasets/UROMOL_BLCA/uromol2_ascat_seg.20200226.txt")
all_genes = read.delim("~/GenomeDK_local/CancerEvolution/phd/Datasets/all_genes_ordered_by_chr.csv", sep = ",")
gene = "LOC105372330"
Chr = 19
gene_start = 22328360 #all_genes[which(all_genes$name2 == gene), "start"] - 1e6
gene_end = 22331580 # all_genes[which(all_genes$name2 == gene), "end"] + 1e6
uromol_ascat$segVal = log2((uromol_ascat$nMajor+uromol_ascat$nMinor)/(uromol_ascat$Ploidy) + 1)
hist(uromol_ascat$segVal)
  
res = data.frame()
for (uromol_id in unique(clinical$SampleID)){ # for all samples
  uromol_ascat_tmp = uromol_ascat %>% filter(sample == uromol_id, chr == Chr)
  if (nrow(uromol_ascat_tmp) >= 1){ # if the sample has ascat calls
    for (i in 1:nrow(uromol_ascat_tmp)){ # for all segments within the sample
      seg_start = uromol_ascat_tmp[i,"startpos"]
      seg_end = uromol_ascat_tmp[i,"endpos"]
      
        if (
            uromol_ascat_tmp[i, "callState"] == "Loss" |
            uromol_ascat_tmp[i, "callState"] == "Del" 
            #uromol_ascat_tmp[i, "nMinor"] == 0 |
            #uromol_ascat_tmp[i, "nMajor"] == 0 
            ){ # if it is a loss
          # if the gene is within the segment
          if(between(gene_start, seg_start, seg_end) || between(gene_end, seg_start, seg_end))
          {
            res <- rbind(res, data.frame(SampleID = uromol_id, impact = "affected"))
          } else {
            res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
          }
        } else {
          res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
        }
      }
  }
}
res = res %>% 
  group_by(SampleID) %>% 
  summarise(impact = sum(impact == "affected") >= 1)
table(res$impact)

tmp = merge(clinical, res, by = "SampleID")
tmp = tmp #%>% filter(Tumor_grade == "High grade")
surv_os <- Surv(as.numeric(as.character(tmp[,'PFS']))/12, as.numeric(as.character(tmp[,'Progression'])))
fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
makeSurvPlot(fit_os,  
             paste0('UROMOL: Loss of ', gene), 
             legen_title = "", 
             ylab = "PFS",
             legend_coord = c(0.1, 0.35),
             colors = MY_COLORS_red_blue)

p_val = surv_pvalue(fit_os)[[2]]
km_res_uromol <- rbind(km_res_uromol, data.frame(gene = gene, pval = p_val, quantcut = 2, data = "Gain", p.adj=0))





#################################################### Mariathasan validation - Load dataset from plots.Rmd #######################################################################################################################################
setwd("~/GenomeDK_local/CancerEvolution/phd/Analysis/Sting-cGAS_Paper")

Total_mariathasan = readRDS("../../Total_mariathasan.rds")
mariathasan_ascat = readRDS("../../Mariathasan_ascat_filtered.rds")
Total_mariathasan$Stage = as.character(Total_mariathasan$TCGA.Subtype)
Total_mariathasan$Met = ifelse(Total_mariathasan$Stage == "IV", "Met", "No Met")

load("../../Datasets/Mariathasan2018/Expression/mariathasan_gene_tpm.20200430.RData")
mariathasan_gene_tpm = log(mariathasan_gene_tpm + 1)
#################################################### GAINS and Loses
gene = "LOC105372330"
gene_start = 22328360 #all_genes[which(all_genes$name2 == gene), "start"] - 1e6
gene_end = 22331580 # all_genes[which(all_genes$name2 == gene), "end"] + 1e6
mariathasan_ascat$segVal = log2((mariathasan_ascat$nAraw+mariathasan_ascat$nBraw)/(mariathasan_ascat$Ploidy) + 1)
hist(mariathasan_ascat$segVal)
mariathasan_ascat$sampleid = as.integer(substr(mariathasan_ascat$ID,6,10))

res = data.frame()
for (uromol_id in unique(Total_mariathasan$subject_id)){ # for all samples
  mariathasan_ascat_tmp = mariathasan_ascat %>% 
    filter(sampleid == uromol_id, 
           Chr == 19)
  
  if (nrow(mariathasan_ascat_tmp) >= 1){ # if the sample has ascat calls
    for (i in 1:nrow(mariathasan_ascat_tmp)){ # for all segments within the sample
      seg_start = mariathasan_ascat_tmp[i,"Start"]
      seg_end = mariathasan_ascat_tmp[i,"End"]
      
      if (mariathasan_ascat_tmp[i, "segVal"] < 1
          #mariathasan_ascat_tmp[i, "callState"] == "Loss" ||
          #mariathasan_ascat_tmp[i, "callState"] == "Del" 
          
      ){ # if it is a loss
        # if the gene is within the segment
        if(between(gene_start, seg_start, seg_end) || between(gene_end, seg_start, seg_end))
        {
          res <- rbind(res, data.frame(SampleID = uromol_id, impact = "affected"))
        } else {
          res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
        }
      } else {
        res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
      }
    }
  }
}
res = res %>% 
  group_by(SampleID) %>% 
  summarise(impact = sum(impact == "affected") >= 1)
table(res$impact)





##################################################################### let's test all gene exp and make forest plot


genes = unlist(all_results %>% 
                 filter(c_type == "BLCA", output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                 dplyr::arrange(desc(scaled_attribtuion)) %>% 
                 head(10) %>% 
                 dplyr::select(gene) )

genes[which(genes == "FAM84B")] <- "LRATD2"
Total_mariathasan$Baseline.ECOG.Score <- as.factor(Total_mariathasan$Baseline.ECOG.Score)


cox_res_mariathasan = data.frame()
wilcox_res_mariathasan = data.frame()
for (g in genes){
  exp = data.frame(gene = mariathasan_gene_tpm[g,])
  exp$SampleID = rownames(exp) 
  tmp = merge(Total_mariathasan, exp, by.x = "ID.x", by.y = "SampleID")
  # Met vs no Met
  tmp = tmp 
  tmp_1 = tmp %>% filter(Met == "Met")
  tmp_2 = tmp %>% filter(Met == "No Met")
  t = compare_means(formula = gene ~ Met,data = tmp )
  t$gene = g
  t$test = "Stage IV vs Stage I,II,III"
  t$ratio = mean(tmp_1$gene)/mean(tmp_2$gene)
  wilcox_res_mariathasan <- rbind(wilcox_res_mariathasan, t)
  # CR vs PD
  tmp = tmp 
  tmp_1 = tmp %>% filter(binaryResponse == "CR/PR")
  tmp_2 = tmp %>% filter(binaryResponse == "SD/PD")
  t = compare_means(formula = gene ~ binaryResponse,data = tmp )
  t$gene = g
  t$ratio = mean(tmp_2$gene)/mean(tmp_1$gene)
  t$test = "SD/PD vs CR/PR"
  wilcox_res_mariathasan <- rbind(wilcox_res_mariathasan, t)
  
  tmp = tmp #%>% filter(Baseline.ECOG.Score >= 1)
  surv_os <- Surv(as.numeric(as.character(tmp[,'os']))/12, as.numeric(as.character(tmp[,'censOS'])))
  cox_model = coxph(formula = surv_os ~ gene + Stage + gender + log10(Neoantigen.burden.per.MB) + Baseline.ECOG.Score, 
                    data = tmp, 
                    robust = T, model = T)
  sum_fit = summary(cox_model)
  for_forest = as.data.frame(sum_fit$coefficients)[1,]
  for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
  #for_forest$param = rownames(for_forest)
  for_forest$gene = g
  cox_res_mariathasan <-rbind(cox_res_mariathasan, for_forest)
}

#wilcox_res_mariathasan$p.adj = p.adjust(wilcox_res_mariathasan$p, method = "fdr")
wilcox_res_mariathasan$gene = unlist(apply(wilcox_res_mariathasan,1, function(x){
  if(as.numeric(x["p.adj"]) > 0.05){
    ""
  } else {
    x["gene"]
  }
}))
wilcox_res_mariathasan$category = c(rep(NA, 8),1, 1, rep(NA,8), 2,2)
ggplot(wilcox_res_mariathasan , aes(x = ratio, y = -log10(p.adj), label = gene, color = test))+
  geom_point() + 
  geom_text_repel()+
  xlab("Ratio")+ 
 # geom_path(aes(group = category)) +
  ggtitle("Mariathasan")+ 
  geom_hline(yintercept = -log10(0.05),color = "red", linetype="dashed" ) +
  geom_vline(xintercept = 1,color = "black", linetype="dashed" ) +
  theme_pubclean() +
  xlim(0.8, 1.1) +
  scale_color_manual(values = c("#1e8479", "#a46750"))


ggplot(wilcox_res_mariathasan, aes(x = gene, y = ratio, fill = test, label = p.signif))+ 
  geom_col(position = "dodge") +
  geom_text() + 
  scale_y_log10()+
  theme_minimal()


#cox_res_mariathasan$adj_p = p.adjust(cox_res_mariathasan$`Pr(>|z|)`, method = "fdr")
cox_res_mariathasan$adj_p = cox_res_mariathasan$`Pr(>|z|)`
breaks = seq(-2,4,by = 0.5)
# and labels
labels = as.character(breaks)
ggplot(data=cox_res_mariathasan,
       aes(x = paste0(gene),y = `coef`, ymin =`X2.5..`, ymax = `X97.5..`))+
  geom_pointrange(aes(col=`exp(coef)`))+
  geom_hline(yintercept =0, linetype=2) + 
  geom_text(aes(label=paste0(signif.num(`adj_p`)),y = 4)) + 
  geom_errorbar(aes(ymin=`X2.5..`,ymax=`X97.5..`,col=`coef`),width=0.5,cex=1)+
  theme(plot.title=element_text(size=16,face="bold"),
        axis.text.x=element_text(face="bold"),
        axis.title=element_text(size=12,face="bold"))+
  
  coord_flip() + 
  theme_pubclean() +
  xlab("") + 
  ggtitle(paste0("Mariathasan Gene Expression")) + 
  scale_color_gradient(guide = F, low = "#000000", high = "#000000") +
  scale_y_continuous(limits = c(-2, 4), breaks = breaks, labels = labels,
                     name = "HR")


################################ 10 random genes
num_sig_random = c()
for(i in 1:100){
  cox_res_rand = data.frame()
  random_genes = sample(rownames(mariathasan_gene_tpm),size = 10)
  for (g in random_genes){
    exp = data.frame(gene = mariathasan_gene_tpm[g,])
    exp$SampleID = rownames(exp) 
    
    # COX 
    tmp = merge(Total_mariathasan, exp, by.x = "ID.x", by.y = "SampleID")
    tmp = tmp %>% filter(Baseline.ECOG.Score >= 1)
    surv_os <- Surv(as.numeric(as.character(tmp[,'os']))/12, as.numeric(as.character(tmp[,'censOS'])))
    cox_model = coxph(formula = surv_os ~  gene + Stage + gender + log10(Neoantigen.burden.per.MB)  +Race  , data = tmp, robust = T, model = T)
    sum_fit = summary(cox_model)
    for_forest = as.data.frame(sum_fit$coefficients)[1,]
    for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
    #for_forest$param = rownames(for_forest)
    for_forest$gene = g
    if (!is.na(for_forest$coef)) {
      cox_res_rand <- rbind(cox_res_rand, for_forest)
    }
  }
  num_sig_random = c(num_sig_random, sum(cox_res_rand$`Pr(>|z|)` < 0.05))
}
median(num_sig_random)

pie_colors = c("#1e8479", "#a5eae2", "#caa394","#a46750" )
pie(c(0.6,0.4,0.9,0.1),
    c("60%","40%", "90%", "10%"),  
    col = pie_colors,
    main = "")
legend(0.9, 0.2, c("Proportion of significant genes Found by GMI","Proportion of not significant genes Found by GMI",
                   "Proportion of not significant genes sampled randomly", "Proportion of significant genes sampled randomly"), cex = 0.8, fill = pie_colors)


matrix(c(0.5, 0.5, 0.9, 0.1),nrow = 2, ncol = 2)
fisher.test()

saveRDS(wilcox_res_mariathasan, "~/Desktop/figure4/mariathasan_gene_Exp_wilcox_binary_response.rds")

############################################## HMF Data ##########################################################################################

hmf_cna = readRDS("~/GenomeDK_local/CancerEvolution/phd/Datasets/HMF/all_genes_CNA.rds"  )
hmf_ascat = readRDS("~/GenomeDK_local/CancerEvolution/phd/Datasets/HMF/HMF_ascat.rds")
hmf_exp = readRDS("~/GenomeDK_local/CancerEvolution/phd/Datasets/HMF/hmf_gex_TPM_matrix.rds")

hmf_tmp = data.frame(hmf_exp)
rownames(hmf_tmp) = rownames(hmf_exp)
hmf_exp = hmf_tmp
hmf_clin = read.delim("~/GenomeDK_local/CancerEvolution/phd/Datasets/HMF/hmf_metadata.csv", sep = ",")
latest_date <- max(as.Date(c(as.character(hmf_clin$biopsyDate), as.character(hmf_clin$deathDate),as.character(hmf_clin$treatmentStartDate), as.character(hmf_clin$treatmentEndDate), as.character(hmf_clin$responseDate)) ), na.rm = T)

HMF_OS1 <-  filter(hmf_clin, deathDate != "NULL") %>%
  mutate(OS = 1, OS.time = as.Date(deathDate)-as.Date(biopsyDate)) 
HMF_OS0 <- filter(hmf_clin, deathDate == "NULL") %>%
  filter(responseMeasured == "Yes") %>% 
  mutate(OS = 0, OS.time = as.Date(latest_date)-as.Date(biopsyDate)) 
Survival_HMF <- bind_rows(HMF_OS1, HMF_OS0) %>% 
  left_join(hmf_clin) 

a = hmf_clin %>% filter(TCGA_type %in% c("BLCA", "COAD", "OV", "STAD", "UCEC", "KIRC"))  

hmf_resp = read.delim("~/GenomeDK_local/CancerEvolution/phd/Datasets/HMF/responses_by_sample 2.tsv")
ids_i_want = hmf_clin[which(hmf_clin$TCGA_type == "KIRC"), "sampleId"]
ids_i_want = intersect(hmf_clin$sampleId, colnames(hmf_exp))

hmf_resp$binaryResponse = case_when(hmf_resp$response == "PR" ~ "PR/CR",
          hmf_resp$response == "CR" ~ "PR/CR",
          hmf_resp$response == "PD" ~ "PD/SD",
          hmf_resp$response == "SD" ~ "PD/SD",
          T ~ ""
)

tmp = data.frame(t(hmf_exp["G6PC", ids_i_want]))
tmp$sampleId = rownames(tmp)
tmp = merge(tmp, hmf_resp, by = "sampleId")

ggplot(tmp %>% filter(binaryResponse != ""), aes(x = binaryResponse, y = G6PC)) + 
  geom_boxplot() + 
  stat_compare_means()+
  theme_minimal()

tmp = merge(tmp, Survival_HMF, by = "sampleId")
tmp = tmp %>% filter(TCGA_type == "KIRC")
tmp$gata6_exp = gtools::quantcut(tmp$G6PC, 2)
surv_os <- Surv(as.numeric(as.character(tmp[,'OS.time']))/365, as.numeric(as.character(tmp[,'OS'])))
fit_os <- survfit(survplot::censor(surv_os, 5)~gata6_exp, data = tmp)
makeSurvPlot(fit_os, 
             paste0('HMF: Expression'), 
             legen_title = "", 
             ylab = "OS",
             legend_coord = c(0.1, 0.35),
             colors = MY_COLORS_red_blue)

library(ggrepel)



total_wilcox_res = data.frame()
total_cox_res = data.frame()
total_km_res = data.frame()


for (c in cancer_types){
  
  cox_res = data.frame()
  wilcox_res = data.frame()
  km_res = data.frame()
  
  genes = unlist(all_results %>% 
                   filter(c_type == c, output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                   dplyr::arrange(desc(scaled_attribtuion)) %>% 
                   head(10) %>% 
                   dplyr::select(gene) )
  for (g in genes){
    if (g == "SNORD111B") {break}
    exp = data.frame(t(hmf_exp[g,]))
    exp$gene = exp[,1]
    exp$sampleId = rownames(exp) 
    # COX 
    tmp = merge(exp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(TCGA_type == c)
    surv_os <- Surv(as.numeric(as.character(tmp[,'OS.time']))/365, as.numeric(as.character(tmp[,'OS'])))
    cox_model = coxph(formula = surv_os ~ gene + treatmentType, data = tmp, robust = T, model = T)
    sum_fit = summary(cox_model)
    for_forest = as.data.frame(sum_fit$coefficients)[1,]
    for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
    for_forest$gene = g
    cox_res <-rbind(cox_res, for_forest)
    
    # KM curves
    # tmp$impact = gtools::quantcut(tmp$gene, 2)
    # fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
    # p_val = surv_pvalue(fit_os)[[2]]
    # km_res <- rbind(km_res, data.frame(gene = g, pval = p_val, quantcut = 2))
    
    
    # Binary Response wilcox
    tmp = merge(exp, hmf_resp, by = "sampleId")
    tmp = merge(tmp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(binaryResponse != "", type == "Chemotherapy", TCGA_type == c)
    cat("Chemo samples : ",nrow(tmp), "\n")
    tmp_1 = tmp %>% filter(binaryResponse == "PR/CR")
    tmp_2 = tmp %>% filter(binaryResponse == "PD/SD")
    if (nrow(tmp_2) & nrow(tmp_1)){
      t = compare_means(formula = gene ~ binaryResponse, data = tmp)
      t$gene = g
      t$ratio = mean(tmp_2$gene)/mean(tmp_1$gene)
      t$therapy = "Chemotherapy"
      wilcox_res <- rbind(wilcox_res, t)
    }

    tmp = merge(exp, hmf_resp, by = "sampleId")
    tmp = merge(tmp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(binaryResponse != "", type == "Immunotherapy",TCGA_type == c)
    cat("Immuno samples : ",nrow(tmp), "\n")
    
    tmp_1 = tmp %>% filter(binaryResponse == "PR/CR")
    tmp_2 = tmp %>% filter(binaryResponse == "PD/SD")
    if (nrow(tmp_2) & nrow(tmp_1)){
      t = compare_means(formula = gene ~ binaryResponse, data = tmp)
      t$gene = g
      t$ratio = mean(tmp_2$gene)/mean(tmp_1$gene)
      t$therapy = "Immunotherapy"
      wilcox_res <- rbind(wilcox_res, t)
    }
   

    tmp = merge(exp, hmf_resp, by = "sampleId")
    tmp = merge(tmp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(binaryResponse != "", type == "Targeted therapy",TCGA_type == c)
    cat("Targeted samples : ",nrow(tmp), "\n")
    tmp_1 = tmp %>% filter(binaryResponse == "PR/CR")
    tmp_2 = tmp %>% filter(binaryResponse == "PD/SD")
    if (nrow(tmp_2) & nrow(tmp_1)){
      t = compare_means(formula = gene ~ binaryResponse, data = tmp)
      t$gene = g
      t$ratio = mean(tmp_2$gene)/mean(tmp_1$gene)
      t$therapy = "Targeted therapy"
      wilcox_res <- rbind(wilcox_res, t)
    }
    
    tmp = merge(exp, hmf_resp, by = "sampleId")
    tmp = merge(tmp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(binaryResponse != "", type == "Multiple therapy",TCGA_type == c)
    cat("Multiple samples : ",nrow(tmp), "\n")
    tmp_1 = tmp %>% filter(binaryResponse == "PR/CR")
    tmp_2 = tmp %>% filter(binaryResponse == "PD/SD")
    if (nrow(tmp_2) & nrow(tmp_1)){
      t = compare_means(formula = gene ~ binaryResponse, data = tmp)
      t$gene = g
      t$ratio = mean(tmp_2$gene)/mean(tmp_1$gene)
      t$therapy = "Multiple therapy"
      wilcox_res <- rbind(wilcox_res, t)
    }
    
    
  }
  
  #wilcox_res$p.adj = p.adjust(wilcox_res$p.adj, method = "fdr")
  #cox_res$p.adj = p.adjust(cox_res$`Pr(>|z|)`, method = "fdr")
  cox_res$p.adj = cox_res$`Pr(>|z|)`
  #km_res$p.adj = p.adjust(km_res$pval, method = "fdr")
  
  wilcox_res$dataset = paste0("HMF - ", c)
  cox_res$dataset = paste0("HMF - ", c)
  #km_res$dataset = paste0("HMF - ", c)

  total_cox_res = rbind(total_cox_res,cox_res )
  total_wilcox_res = rbind(total_wilcox_res, wilcox_res )
  #total_km_res = rbind(total_km_res, km_res )
}

total_wilcox_res$gene = unlist(apply(total_wilcox_res,1, function(x){
  if(as.numeric(x["p.adj"]) < 0.05){
    x["gene"]
  } else {
    ""
  }

}))
ggplot(total_wilcox_res , aes(x = ratio, y = -log10(p.adj), color = dataset, label = gene))+
  geom_point() +
  geom_text_repel(max.overlaps = 50)+
  xlab("Ratio between PD/SD / CR/PR ")+ 
  ggtitle("HMF: Gene expression PD/SP vs CR/PR ") +
  geom_hline(yintercept = -log10(0.05),color = "red", linetype="dashed" ) +
  geom_vline(xintercept = 1,color = "black", linetype="dashed" ) +
  theme_pubclean() + 
  theme(axis.text.x = element_text(angle = 90)) +
  scale_color_manual(values = c("HMF - BLCA"=newCols[6][[1]], "HMF - OV"=newCols[3][[1]], 
                                "HMF - COAD"=newCols[5][[1]], "HMF - STAD"=newCols[2][[1]],
                                "HMF - KIRC"=newCols[4][[1]], "HMF - UCEC"=newCols[1][[1]])) +
  facet_wrap(~therapy, nrow = 1)
total_cox_res$gene = unlist(apply(total_cox_res,1, function(x){
  if(as.numeric(x["p.adj"]) > 0.05){
    ""
  } else {
    x["gene"]
  }
}))
ggplot(total_cox_res, aes(x = coef, y = -log10(p.adj), color = dataset, label = gene))+
  geom_point() + 
  geom_text_repel()+
  xlab("HR")+ 
  ggtitle("HMF: Univariate Cox-PH models")+ 
  geom_hline(yintercept = -log10(0.05),color = "red", linetype="dashed" ) +
  geom_vline(xintercept = 0,color = "black", linetype="dashed" ) +
  theme_minimal() +theme(axis.text.x = element_text(angle = 90)) + 
  scale_color_manual(values = c("HMF - BLCA"=newCols[6][[1]], "HMF - OV"=newCols[3][[1]], 
                                "HMF - COAD"=newCols[5][[1]], "HMF - STAD"=newCols[2][[1]],
                                "HMF - KIRC"=newCols[4][[1]], "HMF - UCEC"=newCols[1][[1]]))
  

ggplot(total_km_res, aes(x = gene, y = -log10(p.adj), color = dataset))+
  geom_point() + 
  xlab("HR")+ 
  ggtitle("HMF: KM curves")+ 
  geom_hline(yintercept = -log10(0.05),color = "red", linetype="dashed" ) +
  geom_vline(xintercept = 0,color = "black", linetype="dashed" ) +
  theme_minimal() +theme(axis.text.x = element_text(angle = 90)) + 
  scale_color_manual(values = c("HMF - BLCA"=newCols[6], "HMF - OV"=newCols[3], 
                                "HMF - COAD"=newCols[5], "HMF - STAD"=newCols[2],
                                "HMF - KIRC"=newCols[4], "HMF - UCEC"=newCols[1]))


total_cox_res = readRDS("~/Desktop/Figure4/hmf_coxph_expression_validation.rds")
total_wilcox_res = readRDS("~/Desktop/Figure4/hmf_expression_validation.rds")

saveRDS(total_wilcox_res, "~/Desktop/Figure4/hmf_expression_validation.rds")
saveRDS(total_cox_res, "~/Desktop/Figure4/hmf_coxph_expression_validation.rds")




# What if we cluster genes in HMF data
library(umap)
library(factoextra)
library(cluster)

ids_i_want = hmf_clin[which(hmf_clin$TCGA_type %in% cancer_types), "sampleId"]
ids_i_want = ids_i_want[ids_i_want %in% colnames(hmf_exp)]
  

total_genes = c()
for (c in cancer_types){
  genes = unlist(all_results %>% 
                   filter(c_type == c, output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                   dplyr::arrange(desc(scaled_attribtuion)) %>% 
                   head(10) %>% 
                   dplyr::select(gene) )
  
  
  for (g in genes){
    if (!g %in% total_genes ){
      total_genes <- c(total_genes, g)
    }
  }
  
}

for_umap = t(hmf_exp[total_genes, ids_i_want])
pc = umap(for_umap, min_dist = 4.9,spread=5, n_neighbors = 50, metric = "cosine") # cosine, correlation, euclidean, manhattan, minkowski
res = as.data.frame(pc$layout)
ggplot(res , aes(x = V1, y = V2)) +
  geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
  theme_pubclean() 
fviz_nbclust(res, pam, method = "silhouette", k.max = 10) + theme_minimal() + ggtitle("The Silhouette Plot")
#gap_stat <- clusGap(res, FUN = pam, nstart = 10, K.max = 10, B = 10)
#fviz_gap_stat(gap_stat) + theme_minimal() + ggtitle("fviz_gap_stat: Gap Statistic")

cl = pam(res, k = 6)
cl = data.frame(cl$clustering)
res$kmeans_cluster = as.factor(cl$cl.clustering)


res$sampleId = rownames(res) 
res = merge(res, hmf_resp %>% filter(sampleId %in% ids_i_want), by = "sampleId")
res = merge(res, hmf_clin %>% filter(sampleId %in% ids_i_want), by = "sampleId")

ggplot(res %>% filter(TCGA_type != "CESC") , aes(x = V1, y = V2, color = binaryResponse)) +
  geom_point() + xlab("UMAP Dim 1")+ ylab("UMAP Dim 1")+  
  theme_pubclean() 



pheatmap(t(for_umap),
         cluster_rows = F,
         clustering_method = "ward.D2",
         show_colnames = F,
          cutree_cols = 6,
         )





# Loooking at gains and losses

chr = 18
hmf_ascat$segVal = log2((hmf_ascat$nAraw + hmf_ascat$nBraw) / (hmf_ascat$Ploidy) + 1)
hist(hmf_ascat$segVal, breaks = 200)
hmf_clin_cancer_type = hmf_clin %>% filter(TCGA_type == "BLCA")
gene = "LOC105372330"
gene_start = 22166898 #all_genes[which(all_genes$name2 == gene), "start"] - 1e6
gene_end = 22168968 # all_genes[which(all_genes$name2 == gene), "end"] + 1e6



res = data.frame()
for (uromol_id in unique(hmf_clin_cancer_type$sampleId)){ # for all samples
  hmf_ascat_tmp = hmf_ascat %>% 
    filter(ID == uromol_id, 
           Chr == chr)
  
  if (nrow(hmf_ascat_tmp) >= 1){ # if the sample has ascat calls
    for (i in 1:nrow(hmf_ascat_tmp)){ # for all segments within the sample
      seg_start = hmf_ascat_tmp[i,"Start"]
      seg_end = hmf_ascat_tmp[i,"End"]
      
      if (hmf_ascat_tmp[i, "segVal"] <= 0.7
      ){ # if it is a loss
        # if the gene is within the segment
        if(between(gene_start, seg_start, seg_end) || between(gene_end, seg_start, seg_end))
        {
          res <- rbind(res, data.frame(SampleID = uromol_id, impact = "affected"))
        } else {
          res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
        }
      } else {
        res <- rbind(res, data.frame(SampleID = uromol_id, impact = "not affected"))
      }
    }
  }
}
res = res %>% 
  group_by(SampleID) %>% 
  summarise(impact = sum(impact == "affected") >= 1)
table(res$impact)

tmp = merge(res, Survival_HMF, by.x = "SampleID",by.y =  "sampleId")
surv_os <- Surv(as.numeric(as.character(tmp[,'OS.time']))/12, as.numeric(as.character(tmp[,'OS'])))
fit_os <- survfit(survplot::censor(surv_os, 5)~impact, data = tmp)
makeSurvPlot(fit_os,  
             paste0('HMF: Loss of ', gene), 
             legen_title = "", 
             ylab = "OS",
             legend_coord = c(0.1, 0.35),
             colors = MY_COLORS_red_blue)



# Validating proportion of genes in cox ph 

med_hmf = median(as.matrix(hmf_exp),na.rm = T)
hmf_exp_filt = hmf_exp[rowMeans(hmf_exp) > med_hmf,]

tt = data.frame()
total_res_prop = data.frame()
for (c in c("KIRC","STAD", "OV", "BLCA", "COAD", "UCEC")){
  cox_res = data.frame()
  genes = unlist(all_results %>% 
                   filter(c_type == c, output_variable == "Metastatic", (type == "Methylation" )) %>%
                   dplyr::arrange(desc(scaled_attribtuion)) %>% 
                   head(10) %>% 
                   dplyr::select(gene) )
  for (g in genes){
    if (g == "SNORD111B") {break}
    exp = data.frame(t(hmf_exp[g,]))
    exp$gene = exp[,1]
    exp$sampleId = rownames(exp) 
    # COX 
    tmp = merge(exp, Survival_HMF, by = "sampleId")
    tmp = tmp %>% filter(TCGA_type == c)
    surv_os <- Surv(as.numeric(as.character(tmp[,'OS.time']))/365, as.numeric(as.character(tmp[,'OS'])))
    cox_model = coxph(formula = surv_os ~ gene + treatmentType + gender + hasSystemicPreTreatment + hasRadiotherapyPreTreatment, data = tmp, robust = T, model = T)
    sum_fit = summary(cox_model)
    for_forest = as.data.frame(sum_fit$coefficients)[1,]
    for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
    for_forest$gene = g
    
    cox_res <-rbind(cox_res, for_forest)
  }
  
  number_of_sign_genes = sum(cox_res$`Pr(>|z|)` < 0.05)
  
  num_sig_random = c()
  for(i in 1:20){
    cox_res_rand = data.frame()
    random_genes = sample(rownames(hmf_exp),size = 10)
    for (g in random_genes){
      exp = data.frame(t(hmf_exp[g,]))
      exp$gene = exp[,1]
      exp$sampleId = rownames(exp) 
      # COX 
      tmp = merge(exp, Survival_HMF, by = "sampleId")
      tmp = tmp %>% filter(TCGA_type == c)
      surv_os <- Surv(as.numeric(as.character(tmp[,'OS.time']))/365, as.numeric(as.character(tmp[,'OS'])))
      cox_model = coxph(formula = surv_os ~ gene + treatmentType + gender + hasSystemicPreTreatment + hasRadiotherapyPreTreatment, data = tmp, robust = T, model = T)
      sum_fit = summary(cox_model)
      for_forest = as.data.frame(sum_fit$coefficients)[1,]
      for_forest = cbind(for_forest, data.frame(confint(cox_model))[1,])
      for_forest$gene = g
      if (!is.na(for_forest$coef)) {
        cox_res_rand <- rbind(cox_res_rand, for_forest)
      } 
      num_sig_random = c(num_sig_random, sum(cox_res_rand$`Pr(>|z|)` < 0.05))
    }
    
    
  }
  
  total_res_prop <- rbind(total_res_prop, data.frame(cancer = c, num_rand_genes = median(num_sig_random), num_picked_genes = number_of_sign_genes, prop = (number_of_sign_genes+1)/(median(num_sig_random)+1)))
  
  
}

total_res_prop

pie_colors2 = c(newCols[4][[1]], newCols[2][[1]], newCols[3][[1]], newCols[6][[1]], newCols[5][[1]], newCols[1][[1]])
pie(c(2/1,2/1,4/1,0.05,1,3/1),
    c("2/1","2/1", "4/1", "0/0", "1/0", "3/1"),  
    col = pie_colors2,
    main = "")
legend(0.9, 0.2, c("KIRC", "STAD", "OV", "BLCA", "COAD", "UCEC"), cex = 0.8, fill = pie_colors2)

##################################### PRECOG #########################################
precog = read.delim("~/GenomeDK_local/CancerEvolution/phd/Analysis/PRECOG-metaZ.pcl_fix.txt")

precog_res = data.frame()
for (c in c("KIRC","STAD", "OV", "BLCA", "COAD")){
  
  genes = unlist(all_results %>% 
                   filter(c_type == c, output_variable == "Metastatic", (type == "Expression" | type == "Methylation")) %>%
                   dplyr::arrange(desc(scaled_attribtuion)) %>% 
                   head(10) %>% 
                   dplyr::select(gene) )
  for (g in genes){
    if (g == "SNORD111B" | g == "SMIM15") {break}
    if (c == "OV"){can_type = "Ovarian_cancer"} else if(c == "BLCA"){can_type = "Bladder_cancer"} else if(c == "COAD"){can_type = "Colon_cancer"} else if(c == "KIRC"){can_type = "Kidney_cancer"} else if (c == "STAD"){can_type = "Gastric_cancer"}
    
    precog_res <- rbind(precog_res, data.frame(gene = g, zscore=precog[which(precog$Gene == g), can_type], cancer_type = c))
  }
}

ggplot(precog_res, aes(x = gene, y = zscore, color = cancer_type)) +
  geom_point() + 
  geom_hline(yintercept = -1.96)+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))

ggplot(precog %>% filter(Gene %in% genes), aes(x = Gene, y = Unweighted_meta.Z_of_all_cancers)) +
  geom_point() + 
  geom_hline(yintercept = -1.96)+
  geom_hline(yintercept = 1.96)+
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90))








publications = read.delim("~/Desktop/annual-number-of-ai-publications.csv", sep = ",")
ggplot(publications, aes(x = Year, y = Number.of.AI.Publications)) +
  geom_point()+
  geom_line()+
  theme_minimal()


f_x <- function(x) {
  return( cos((x^2)/x) + sin(5*x)) 
}
rand = data.frame(x = 1:20, y = f_x(1:20))
ggplot(rand, aes(x=x, y=y)) +
  geom_line() + 
  xlab("X") + ylab("Y")+
  theme_minimal()
