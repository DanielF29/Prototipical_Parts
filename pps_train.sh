# Trainings of ProtoPNet, OsDA
##############################################################################################################################################
#####PPs = 1  ################################################################################################################################
#run_num=${$2}
#Renet50 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=1 -num_classes=6 -experiment_run=001 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=1 -num_classes=6 -experiment_run=002 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=1 -num_classes=6 -experiment_run=003 -run=$2
#
#densenet201 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=1 -num_classes=6 -experiment_run=004 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=1 -num_classes=6 -experiment_run=005 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=1 -num_classes=6 -experiment_run=006 -run=$2
#
#vgg16 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=1 -num_classes=6 -experiment_run=007 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=1 -num_classes=6 -experiment_run=008 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=1 -num_classes=6 -experiment_run=009 -run=$2
########################################################################################################################################
#####PPs = 3  ##########################################################################################################################
#Renet50 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=3 -num_classes=6 -experiment_run=010 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=3 -num_classes=6 -experiment_run=011 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=3 -num_classes=6 -experiment_run=012 -run=$2
#
#densenet201 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=3 -num_classes=6 -experiment_run=013 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=3 -num_classes=6 -experiment_run=014 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=3 -num_classes=6 -experiment_run=015 -run=$2
#
#vgg16 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=3 -num_classes=6 -experiment_run=016 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=3 -num_classes=6 -experiment_run=017 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=3 -num_classes=6 -experiment_run=018 -run=$2
########################################################################################################################################
#####PPs = 10  #########################################################################################################################
#Renet50 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=10 -num_classes=6 -experiment_run=019 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=10 -num_classes=6 -experiment_run=020 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=10 -num_classes=6 -experiment_run=021 -run=$2
#
#densenet201 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=10 -num_classes=6 -experiment_run=022 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=10 -num_classes=6 -experiment_run=023 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=10 -num_classes=6 -experiment_run=024 -run=$2
#
#vgg16 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=10 -num_classes=6 -experiment_run=025 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=10 -num_classes=6 -experiment_run=026 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=10 -num_classes=6 -experiment_run=027 -run=$2
########################################################################################################################################
#####PPs = 50   ########################################################################################################################
#Renet50 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=50 -num_classes=6 -experiment_run=028 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=50 -num_classes=6 -experiment_run=029 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=50 -num_classes=6 -experiment_run=030 -run=$2
#
#densenet201 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=50 -num_classes=6 -experiment_run=031 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=50 -num_classes=6 -experiment_run=032 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=50 -num_classes=6 -experiment_run=033 -run=$2
#
#vgg16 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=50 -num_classes=6 -experiment_run=034 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=50 -num_classes=6 -experiment_run=035 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=50 -num_classes=6 -experiment_run=036 -run=$2
####################################################################################################################################
#####PPs = 100 #####################################################################################################################
#Renet50 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=100 -num_classes=6 -experiment_run=037 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=100 -num_classes=6 -experiment_run=038 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=resnet50 -pps_per_class=100 -num_classes=6 -experiment_run=039 -run=$2
#
#densenet201 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=100 -num_classes=6 -experiment_run=040 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=100 -num_classes=6 -experiment_run=041 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=densenet201 -pps_per_class=100 -num_classes=6 -experiment_run=042 -run=$2
#
#vgg16 trainings
python3 mainSection_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=100 -num_classes=6 -experiment_run=043 -run=$2
python3 mainSurface_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=100 -num_classes=6 -experiment_run=044 -run=$2
python3 mainMix_OsDA.py -gpuid=$1 -base_architecture=vgg16 -pps_per_class=100 -num_classes=6 -experiment_run=045 -run=$2
#
