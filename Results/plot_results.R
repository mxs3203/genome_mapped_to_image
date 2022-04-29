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

setwd("~/pytorch_docker/TCGA_GenomeImage/Results")
vega = read.delim2("sanchez-vega-pws_1026.csv", sep = ";")
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
cancer_types = c("UCEC", "STAD", "OV", "KIRC", "COAD", "BLCA")
scenarios = c("Age","CancerType","Metastatic", "TP53", "wGII")
data_sources = c("Methylation","Expression","Gain","Loss","Mutation")
top_n = 50
#saveRDS(all_results, "all_results_merged_ready_for_analysis.rds")

all_results = readRDS("all_results_merged_ready_for_analysis.rds")

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
    tmp_data = read.csv(paste0(s,"/Squere/",type,"_SquereImg_",s,"_top_",num_res,".csv"))
    colnames(tmp_data)= c("rowname","attribution", "type","gene")
    tmp_data$c_type = type
    tmp_data$top_n = NA
    tmp_data$output_variable = s
    
    t = tmp_data %>% filter(type == "Expression") %>% arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_exp = rbind(data_exp, t)
    
    t = tmp_data %>% filter(type == "Gain") %>% arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_gain = rbind(data_gain, t)
    
    t = tmp_data %>% filter(type == "Loss") %>% arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_loss = rbind(data_loss, t)
    
    t = tmp_data %>% filter(type == "Mutation") %>% arrange(desc(attribution))
    t[1:top_n, "top_n"] = T
    data_mut = rbind(data_mut, t)
    
    t = tmp_data %>% filter(type == "Methylation") %>% arrange(desc(attribution))
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
for (d_s in data_sources){
  tmp = all_results %>% filter(type == d_s)
  tmp = tmp %>% mutate(scaled_attribtuion = range01(attribution)) 
  tmp_tmp = rbind(tmp_tmp, tmp)
}
nrow(all_results) == nrow(tmp_tmp)
all_results = tmp_tmp
############## END OF Mandatory reading of all data

mean(all_results$scaled_attribtuion)
############### Figure 3 summarizing sources and cancer types


summ_res = all_results %>% 
  filter(top_n) %>%
  group_by(type, c_type,output_variable) %>%
  summarize(m = mean(scaled_attribtuion)) 

ggplot(summ_res, aes(x =output_variable , y = m, color = type)) + 
  geom_point()+
  scale_y_log10()+
  ylab("log10(Att)")+
  coord_flip()+
  theme_pubclean()+ 
  facet_wrap(~c_type) 


  
############## For HUGE heatmap containting multiple heatmaps
all = data.frame()
cnt = 1
for (t in unique(all_results$type)) { # for every data source
  for (c in unique(all_results$c_type)) { # for every c type
    cols = c()
    rows = c()
    nums = c()
    for (var in unique(all_results$output_variable)) { # for every output variable
      var = "Metastatic"
      tmp = all_results %>% filter(c_type == c, 
                                   top_n == T, 
                                   type == t,
                                   output_variable == var)
      nums = c(nums, tmp$scaled_attribtuion)
      cols = c(cols, tmp$output_variable)
      rows = c(rows, as.character(tmp$gene))
      break
    }
    
    d = data.frame(nums = nums, vars = cols, genes = rows, c_type = c, type = t)
    if (t %in% c("Gain","Loss")){
      fontSize = 5
    } else {
      fontSize = 6
    }
    all = rbind(all, d)
    assign(x=paste0("p",cnt),value = ggplot(d, aes(reorder(genes, -nums), vars, fill = nums)) +
      geom_tile() +
      scale_fill_gradient2(midpoint = 0.004,
                           low = "#02A676", mid = "#007369", high = "#003840") + 
      theme_minimal() +
      xlab("")+ ylab("") +
      #ggtitle(paste(c,t,var))+
      theme(legend.title = element_blank(),
            #legend.position = "none",
            title = element_text(size = 5),
            axis.text.x = element_text(size = fontSize, angle=90,hjust=1,vjust=0.7),
            axis.text.y = element_blank(),
            plot.margin = unit(c(0, 0, 0, 0), "cm")) +
      coord_equal() )
    
    cnt = cnt + 1
  }
}
end = (cnt-1)
ggsave(filename = "~/Desktop/p.png", width = 17, height =9,
       plot = cowplot::plot_grid(plotlist = 
                                   mget(paste0("p", 1:end)), 
                                 nrow = 6, ncol = 5))


  #ggtitle(paste(c,t,var))+
  theme(legend.title = element_blank(),
        #legend.position = "none",
        title = element_text(size = 5),
        axis.text.x = element_text(size = fontSize, angle=90,hjust=1,vjust=0.7),
        axis.text.y = element_blank(),
        plot.margin = unit(c(0, 0, 0, 0), "cm")) +
  coord_equal() 


  ggplot(all, aes(x = c_type, y = nums, color = type, label = genes)) +
  geom_point() + 
  geom_text_repel() +
  theme_pubclean()
############## END OF  For HUGE heatmap containting multiple heatmaps
  
  
############## Reactome Pathways
quantile(all_results$scaled_attribtuion, probs = c(0.10, 0.90, 0.95, 0.99))
  
library(ReactomePA)
reactome_res = data.frame()
for (c in cancer_types){
  for (t in data_sources) {
    print(paste0(c, t))
    tmp = all_results %>% dplyr::filter(output_variable == "Age", 
                                        c_type == c,
                                        type == t) %>%
      dplyr::arrange(desc(scaled_attribtuion)) %>% 
      head(100) #%>% 
      #filter(scaled_attribtuion >=  0.07278) # 
    if(nrow(tmp) > 0){
      mapping = map2entrez(as.character(tmp$gene))
      tmp = merge(tmp, mapping, by = "gene", all.x= T)
      x <- enrichPathway(gene=tmp$enterez, 
                         pvalueCutoff=0.05,
                         pAdjustMethod = "bonferroni",
                         readable=T)
      
      if(!is.null(x)){
        x = x@result
        x = x %>% filter(p.adjust < 0.05) 
        if(nrow(x) > 0){
          x$c_type = c
          x$type = t
          reactome_res <- rbind(reactome_res, data.frame(x))
        }
      }
     
    }
   
  }
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




############## Encoded genomes UMAP #########
library(umap)

metadata = read.csv("main_meta_data.csv")
for (param in scenarios){
  tmp_data = read.csv(paste0("Results/",param,"/encoded_genomes.csv"))
  var = case_when(
    param == "TP53" ~ "tp53",
    param  == "wGII" ~ "wGII",
    param  == "Metastatic" ~ "met",
    param  == "Age" ~ "age_at_initial_pathologic_diagnosis",
    param  == "CancerType" ~ "type",
    TRUE ~ as.character(param)
  )
  tmp_data = merge(metadata %>% 
                     #dplyr::filter(type == "KIRC") %>%
                     dplyr::select("id", var) , tmp_data, by.x="id", by.y = "sampleid")
  umap_res = umap(tmp_data[, paste0("X", 0:127)], min_dist=0.9)
  res = data.frame(umap_res$layout)
  res$var = tmp_data[,var]
  assign(x= paste0("p_",param), value = ggplot(res, aes(x = X1, y = X2, color = var)) +
    geom_point() + 
    ggtitle(param) +
    theme_bw())
}

cowplot::plot_grid(p_wGII, p_Metastatic, p_TP53, p_CancerType, p_Age, labels = "auto")

############## END OF Encoded genomes UMAP #########

library(RColorBrewer)

heatmaps = list()
cnt = 1
for (s in c("Methylation","Expression","Mutation")){
    source = s
    tmp = all_results %>% filter(top_n == T, 
                                 type == source,
                                 output_variable == "Age")
    for_heatmap = as.data.frame(matrix(nrow = length(unique(tmp$gene)), ncol = length(cancer_types)))
    
    for (i in 1:length(unique(tmp$gene))){
      for (j in 1:length(cancer_types)){
        tt = tmp %>% filter(tmp$gene == unique(tmp$gene)[i], c_type == cancer_types[j] )
        if (nrow(tt) == 0){
          for_heatmap[i,j] = 0
        } else {
          for_heatmap[i,j] = mean(tt$scaled_attribtuion)
        }
      }
    }
    colnames(for_heatmap) = cancer_types
    rownames(for_heatmap) = unique(tmp$gene)
    
    coul <- colorRampPalette(brewer.pal(8, "Greens"))(6)
    heatmaps[[cnt]] <- as_ggplot( pheatmap(for_heatmap,
             cluster_rows = F, 
             cluster_cols = T, 
             main = source,
             scale = "row",
             cellwidth = 15,
             cellheight = 15,
             clustering_method = "ward.D2",
             col = coul,
             fontsize = 8)[[4]] ) + theme(plot.margin = unit(c(0, 0, 0, 0), "cm"))
    cnt = cnt + 1
}

ggarrange(heatmaps[[1]],
          heatmaps[[2]],
          heatmaps[[3]],
          ncol = 3,
          common.legend = T,
          align = 'v')




# Sanchez Vega Grouping 
tmp = all_results %>% filter(c_type == "KIRC", output_variable == "Metastatic")


results = data.frame()
for (i in 1:length(unique(vega$Pathway_pretty))) {
  vega_group = vega %>% filter(Pathway_pretty == unique(vega$Pathway_pretty)[i]) 
  for (j in 1:length(unique(tmp$type))){
    dlbc_tmp = tmp %>% filter(type == unique(tmp$type)[j])
    num = sum(dlbc_tmp[which(dlbc_tmp$gene %in% vega_group$Gene), "scaled_attribtuion"])
    results[i,j] = num
  }
}
colnames(results) = unique(tmp$type)
rownames(results) = unique(vega$Pathway_pretty)

pheatmap(results,
         cluster_rows = T,
         cluster_cols = T,
         scale="column")

# Immune system grouping

immune_genes <- read.table("selected_markers.txt", sep = '\t', header = TRUE, as.is = TRUE)
immune_gene_groups <- unique(immune_genes$Cell.Type)


results = data.frame()
for (i in 1:length(immune_gene_groups)) {
  group = immune_genes %>% filter(Cell.Type == unique(immune_genes$Cell.Type)[i]) 
  for (j in 1:length(unique(tmp$type))){
    dlbc_tmp = tmp %>% filter(type == unique(tmp$type)[j])
    num = sum(dlbc_tmp[which(dlbc_tmp$gene %in% group$Gene), "attribution"])
    results[i,j] = num
  }
}
colnames(results) = unique(tmp$type)
rownames(results) = unique(immune_genes$Cell.Type)

pheatmap(results,
         cluster_rows = F,
         cluster_cols = F, 
         scale = "column")


# Hallmarks grouping
library(msigdbr)
m_df = msigdbr(species = "Homo sapiens", category = "H")
m_df %>% dplyr::group_by(gs_name) %>% dplyr::summarise(list(human_gene_symbol))


results = data.frame()
for (i in 1:length(unique(m_df$gs_name))) {
  group = m_df %>% filter(gs_name == unique(m_df$gs_name)[i]) 
  for (j in 1:length(unique(tmp$type))){
    dlbc_tmp = tmp %>% filter(type == unique(tmp$type)[j])
    num = sum(dlbc_tmp[which(dlbc_tmp$gene %in% group$gene_symbol), "attribution"])
    results[i,j] = num
  }
}
colnames(results) = unique(tmp$type)
rownames(results) = unique(m_df$gs_name)

pheatmap(results,
         cluster_rows = F,
         cluster_cols = F, 
         scale = "column",
         fontsize = 6,
         cellwidth = 5)


################## # All Cancer Types together##############

files = list.files("data/TP53/24x3760Image/")
ind = endsWith(files, suffix = "38000.csv")

total = data.frame()
for (file in files[ind]){
  type = substr(file, start = 3, stop = 6)
  tmp = read.csv(paste0("data/TP53/24x3760Image/",file))  
  tmp$cancer_type = type
  total <- rbind(total, tmp)
}
colnames(total) <- c("X", "attribution", "type", "gene","cancer_type")

ggplot(total, aes(x = cancer_type, y = attribution, fill = type))+
  geom_col(position = "dodge") + 
  theme_pubclean()


total_vega = merge(total, vega, by.x ="gene", by.y = "Gene")
ggplot(total_vega, aes(x = cancer_type, y = attribution, fill = type))+
  geom_col(position = "dodge") + 
  theme_pubclean() + 
  facet_wrap(~Pathway, nrow = 5, ncol = 2)





