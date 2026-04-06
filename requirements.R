# Install CRAN packages
install.packages(c(
  "imager",
  "randomForest",
  "reshape2",
  "pROC",
  "tidyr",
  "ggplot2",
  "caret",
  "moments"
))

# Install Bioconductor package
if (!require("BiocManager")) {
  install.packages("BiocManager")
}

if (!require("EBImage")) {
  BiocManager::install("EBImage")
}