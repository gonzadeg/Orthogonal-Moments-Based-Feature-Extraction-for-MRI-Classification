##########################################################################################
#0) CARGAR LIBRERÍAS NECESARIAS
##########################################################################################

if(!require("imager")) {
  install.packages("imager")
  library("imager")}

if(!require("randomForest")) {
  install.packages("randomForest")
  library("randomForest")}

if(!require("reshape2")) {
  install.packages("reshape2")
  library("reshape2")}

if(!require("pROC")) {
  install.packages("pROC")
  library("pROC")}

if(!require("tidyr")) {
  install.packages("tidyr")
  library("tidyr")}

if(!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")}

if(!require("caret")) {
  install.packages("caret")
  library("caret")}

if(!require("moments")) {
  install.packages("moments")
  library("moments")}

if(!require(EBImage)) BiocManager::install("EBImage")

##########################################################################################
#1) FUNCIONES RELEVANTES
##########################################################################################

# Calcular polinomios de Chebyshev (II)
chebyshev_second_type = function(n, x) {
  if (n == 0) return(rep(1, length(x)))
  if (n == 1) return(2 * x)            
  
  Un_2 = rep(1, length(x))
  Un_1 = 2 * x            
  
  for (k in 2:n) {
    Un = 2 * x * Un_1 - Un_2
    Un_2 = Un_1
    Un_1 = Un
  }
  return(Un_1)
}

# Calcular los momentos de Chebyshev (II)
calculate_chebyshev_second_moment = function(img_vector, p, q, nrow, ncol) {
  m = as.numeric(nrow)
  n = as.numeric(ncol)
  
  # Generar secuencias
  x = seq(-1, 1, length.out = m)
  y = seq(-1, 1, length.out = n)
  
  # Calcular polinomios
  Up = chebyshev_second_type(p, x)
  Uq = chebyshev_second_type(q, y)
  
  # Reconstrucción de imagen
  img_matrix = matrix(img_vector, nrow = m, ncol = n)
  
  # Calcular el momento de Chebyshev
  CH2pq = img_matrix * (Up %*% t(Uq))
  moment = sum(CH2pq) * (2 / m) * (2 / n)
  return(moment)
}


# Función para procesar un batch de imágenes
process_batch = function(image_paths, class_name, max_dim) {
  images = lapply(image_paths, load.image)
  
  images_resized = lapply(images, function(img) {
    imager::resize(img, size_x = max_dim, size_y = max_dim, interpolation_type = 3)
  })
  features = lapply(images_resized, function(img) {
    if (dim(img)[3] == 3) {
      img_gray = grayscale(img)
    } else {
      img_gray = img
    }
    
    # Imagen a vector
    img_array  = as.array(img_gray)
    img_vector = as.vector(img_array)
    
    # Validación y ajuste del vector
    expected_length = dim(img_array)[1] * dim(img_array)[2]
    if (length(img_vector) > expected_length) {
      img_vector = img_vector[1:expected_length]
    }
    
    nrow = dim(img_array)[1]
    ncol = dim(img_array)[2]
    
    # Calcular los momentos de Chebyshev (II)
    p_max            = 12
    q_max            = 12
    chebyshev2_moments = matrix(0, nrow = p_max + 1, ncol = q_max + 1)
    
    for (p in 0:p_max) {
      for (q in 0:q_max) {
        chebyshev2_moments[p + 1, q + 1] = calculate_chebyshev_second_moment(
          img_vector, p, q, nrow = nrow, ncol = ncol
        )
      }
    }
    
    # Momentos a vector de características
    chebyshev2_features        = as.vector(chebyshev2_moments)
    names(chebyshev2_features) = paste0("DCM2_p", 0:p_max, "_q", 0:q_max)
    return(chebyshev2_features)
  })
  
  
  # DF del batch
  features_df       = data.frame(do.call(rbind, features))
  features_df$class = class_name
  return(features_df)
}

# Dividir imágenes en batches
split_batches = function(files, batch_size) {
  split(files, ceiling(seq_along(files) / batch_size))
}

# Procesamiento en batches por clase
process_images_batches = function(image_paths, class_name, batch_size) {
  batches = split_batches(image_paths, batch_size)
  max_dim = max(sapply(image_paths, function(f) max(dim(load.image(f)))))
  do.call(rbind, lapply(batches, process_batch, class_name = class_name, max_dim = max_dim))
}

# Cantidad de imágenes por clase (1 batch)
batch_size = 200

##########################################################################################
#3) PREPARACIÓN DE DATOS DE TRAINING Y TESTING
##########################################################################################

# ACCESO A IMÁGENES ######################################################################

no_tumor_folder   = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Training\\notumor"    #class 1
glioma_folder     = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Training\\glioma"     #class 2
pituitary_folder  = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Training\\pituitary"  #class 3
meningioma_folder = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Training\\meningioma" #class 4

pattern           = ".jpg"

files_normal      = list.files(no_tumor_folder,   pattern = pattern, full.names = TRUE)
files_glioma      = list.files(glioma_folder,     pattern = pattern, full.names = TRUE)
files_pituitary   = list.files(pituitary_folder,  pattern = pattern, full.names = TRUE)
files_meningioma  = list.files(meningioma_folder, pattern = pattern, full.names = TRUE)

testing_no_tumor_folder   = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Testing\\notumor"
testing_glioma_folder     = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Testing\\glioma"
testing_pituitary_folder  = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Testing\\pituitary"
testing_meningioma_folder = "C:\\Users\\gonza\\Documents\\Degiuseppe3\\Testing\\meningioma"

testing_files_normal      = list.files(testing_no_tumor_folder, pattern = pattern, full.names = TRUE)
testing_files_glioma      = list.files(testing_glioma_folder, pattern = pattern, full.names = TRUE)
testing_files_pituitary   = list.files(testing_pituitary_folder, pattern = pattern, full.names = TRUE)
testing_files_meningioma  = list.files(testing_meningioma_folder, pattern = pattern, full.names = TRUE)


# PROCESAMIENTO DE IMÁGENES POR CLASE ####################################################

features_normal_df     = process_images_batches(files_normal, "No_Tumor", batch_size)
features_glioma_df     = process_images_batches(files_glioma, "Glioma", batch_size)
features_pituitary_df  = process_images_batches(files_pituitary, "Pituitary", batch_size)
features_meningioma_df = process_images_batches(files_meningioma, "Meningioma", batch_size)

train_data       = rbind(features_normal_df, features_glioma_df, features_pituitary_df, features_meningioma_df)
train_data$class = factor(train_data$class, levels = c("No_Tumor", "Glioma", "Pituitary", "Meningioma"))

tefeatures_normal_df     = process_images_batches(testing_files_normal, "No_Tumor", batch_size)
tefeatures_glioma_df     = process_images_batches(testing_files_glioma, "Glioma", batch_size)
tefeatures_pituitary_df  = process_images_batches(testing_files_pituitary, "Pituitary", batch_size)
tefeatures_meningioma_df = process_images_batches(testing_files_meningioma, "Meningioma", batch_size)

test_data       = rbind(tefeatures_normal_df, tefeatures_glioma_df, tefeatures_pituitary_df, tefeatures_meningioma_df)
test_data$class = factor(test_data$class, levels = c("No_Tumor", "Glioma", "Pituitary", "Meningioma"))

##########################################################################################
#4) MODELO
##########################################################################################

# Modelo Random Forest
set.seed(123)
rf_model = randomForest(
  class ~ .,
  data = train_data,
  importance = TRUE
)

importance_matrix = importance(rf_model)
print("Importancia de características inicial:")
print(importance_matrix)

# SELECCIÓN DE CARACTERÍSTICAS ###########################################################

# Ordenar características por importancia y elegir las más útiles
important_features = rownames(importance_matrix[order(-importance_matrix[, 1]),])
top_n              = 75
selected_features  = important_features[1:top_n]

# Crear datasets con las características seleccionadas
train_data_filtered = train_data[, c(selected_features, "class")]
test_data_filtered  = test_data[, c(selected_features, "class")]

# Entrenar el modelo con las características seleccionadas 
set.seed(123)
rf_model = randomForest(
  class ~ .,
  data = train_data_filtered,
  importance = TRUE
)

print(rf_model)

rf_predictions = predict(rf_model, newdata = test_data_filtered)

# Matriz de confusión
confusion_matrix  = confusionMatrix(rf_predictions, test_data_filtered$class)
print(confusion_matrix)
importance_matrix = importance(rf_model)
print("Importancia de características inicial:")
print(importance_matrix)

##########################################################################################
# 5) MÉTRICAS DE RENDIMIENTO
##########################################################################################

# Tabla de confusión como matriz
confusion_table = confusion_matrix$table

# Vectores para métricas
f1_scores          = c()
tpr_values         = c()
precision_values   = c()
specificity_values = c()

# Métricas para cada clase
for (class in levels(test_data_filtered$class)) {
  TP = confusion_table[class, class]
  FP = sum(confusion_table[class, ]) - TP
  FN = sum(confusion_table[, class]) - TP
  TN = sum(confusion_table) - (TP + FP + FN)
  
  # Calcular métricas
  precision   = ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  recall      = ifelse((TP + FN) > 0, TP / (TP + FN), 0)
  specificity = ifelse((TN + FP) > 0, TN / (TN + FP), 0)
  f1_score    = ifelse((precision + recall) > 0, 2 * (precision * recall) / (precision + recall), 0)
  
  # Guardar métricas
  precision_values   = c(precision_values, precision)
  tpr_values         = c(tpr_values, recall)
  specificity_values = c(specificity_values, specificity)
  f1_scores          = c(f1_scores, f1_score)
}

# Métricas generales del modelo
accuracy          = sum(diag(confusion_table)) / sum(confusion_table)
macro_f1          = mean(f1_scores)
weighted_f1       = sum(f1_scores * colSums(confusion_table) / sum(confusion_table))
balanced_accuracy = mean((tpr_values + specificity_values) / 2)

# DF con métricas por clase
metrics_by_class = data.frame(
  Class       = levels(test_data_filtered$class),
  Precision   = precision_values,
  TPR         = tpr_values,
  Specificity = specificity_values,
  F1_Score    = f1_scores
)

print("Métricas por clase:")
print(metrics_by_class)

general_metrics = data.frame(
  Metric = c("Accuracy", "Macro F1 Score", "Weighted F1 Score", "Balanced Accuracy"),
  Value  = c(accuracy, macro_f1, weighted_f1, balanced_accuracy)
)

print("Métricas generales del modelo:")
print(general_metrics)

##########################################################################################
# 6) P-VALUES E INTERVALOS DE CONFIANZA DE MOMENTOS SELECCIONADOS
##########################################################################################

# LIBRERIAS
if(!require("dplyr")) install.packages("dplyr")
if(!require("writexl")) install.packages("writexl")
library(dplyr)
library(writexl)

# Dataset solo con las features seleccionadas y clase
features_df = train_data[, c(selected_features, "class")]

# 1. Calcular p-values por ANOVA y Kruskal-Wallis para cada momento (feature)
anova_pvals = sapply(selected_features, function(feat) {
  fit = aov(features_df[[feat]] ~ features_df$class)
  summary(fit)[[1]][["Pr(>F)"]][1]
})
anova_pvals_df = data.frame(
  Feature = selected_features,
  ANOVA_pvalue = anova_pvals,
  row.names = NULL
)

kw_pvals = sapply(selected_features, function(feat) {
  kruskal.test(features_df[[feat]], features_df$class)$p.value
})
kw_pvals_df = data.frame(
  Feature = selected_features,
  Kruskal_Wallis_pvalue = kw_pvals,
  row.names = NULL
)

# Unir ambas tablas de p-values
pvals_table = merge(anova_pvals_df, kw_pvals_df, by = "Feature")

# 2. Calcular intervalos de confianza (IC 95%) por feature y clase
ci_table_global = data.frame()

for (feature in selected_features) {
  values  = features_df[[feature]]
  m       = mean(values)
  s       = sd(values)
  var_val = var(values)
  min_val = min(values)
  max_val = max(values)
  n       = length(values)
  error   = qt(0.975, df = n - 1) * s / sqrt(n)  # IC 95%
  
  ci_table_global = rbind(ci_table_global, data.frame(
    Feature   = feature,
    Mean      = m,
    SD        = s,
    Variance  = var_val,
    Min       = min_val,
    Max       = max_val,
    N         = n,
    Lower_CI  = m - error,
    Upper_CI  = m + error
  ))
}


# 4. Imprimir una muestra de resultados
cat("Ejemplo de p-values para los momentos seleccionados:\n")
print(head(pvals_table, 10))

cat("\nEjemplo de intervalos de confianza para los momentos seleccionados (primeros 10):\n")
print(head(ci_table_global, 10))

##########################################################################################
#7) GRÁFICOS
##########################################################################################

# MATRIZ DE CONFUSIÓN ####################################################################

#confusion_data           = as.data.frame(confusion_matrix$table)
#colnames(confusion_data) = c("Prediction", "Reference", "Frequency")

#ggplot(confusion_data, aes(x = Reference, y = Prediction)) +
#geom_tile(aes(fill = Frequency), color = "gray") +
#geom_text(aes(label = Frequency), vjust = 1) +
#scale_fill_gradient(low = "white", high = "darkgray") +
#labs(x = "True Class", y = "Predicted Class") +
#theme_minimal()

# METRICAS POR CLASE (RELEVANTES) ########################################################

#metrics       = as.data.frame(confusion_matrix$byClass)
#metrics$class = rownames(metrics)
#metrics_plot  = metrics[, c("class", "Sensitivity", "Specificity", "Precision", "Balanced Accuracy")]

# CURVA ROC y AUC ########################################################################

#rf_probabilities = predict(rf_model, newdata = test_data, type = "prob")

# ROC para cada clase
#roc_curves       = lapply(levels(test_data$class), function(cls) {
#roc(as.numeric(test_data$class == cls), rf_probabilities[, cls], quiet = TRUE)
})

#plot(roc_curves[[1]], col = "red", lwd = 2)
#for (i in 2:length(roc_curves)) {
#plot(roc_curves[[i]], col = i + 1, lwd = 2, add = TRUE)
}
#legend("bottomright", legend = levels(test_data$class), col = 2:(length(roc_curves) + 1), lwd = 2)

# AUC para cada clase
#auc_values = sapply(roc_curves, function(roc_curve) auc(roc_curve))
#auc_df     = data.frame(Class = levels(test_data$class), AUC = auc_values)
#print(auc_df)