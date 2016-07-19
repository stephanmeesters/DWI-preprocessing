
'''
    Processing pipeline for DW-MRI
    Author: Stephan Meesters
    2016
'''

# use local version of NiPype
import os, sys
sys.path.append(os.path.abspath('nipype'))

# Define inputs
# Single subject
dataDirectory = '<path to data directory>'
outputDirectory = '<path to output directory>'
t1_file = '<path to T1 NiFTI file relative to dataDirectory>'
dwiDirectories = ['<path to diffusion data directory relative to dataDirectory'>]
rois_files = ['ROIs/lgn_left.nii','ROIs/v1_left.nii','ROIs/lgn_right.nii','ROIs/v1_right.nii','ROIs/exclude_left.nii','ROIs/exclude_right.nii']
rois_files = [dataDirectory+x for x in rois_files]

# Settings
flip_bvals_x = True
perform_tracking = True
perform_fbc = True

#-------------#

# Load packages
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.fsl as fsl          # fsl
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import os                                    # system functions
import procfunc as func

# config
from nipype import config
cfg = dict(logging=dict(workflow_level = 'DEBUG'),
           execution={'stop_on_first_crash': False,
                      'hash_method': 'timestamp'}) # content
config.update_config(cfg)

# Info source
info = dict(dwi=[['dwi_dirs']],
            bvecs=[['dwi_dirs']],
            bvals=[['dwi_dirs']])

infosource = pe.Node(interface=util.IdentityInterface(fields=['dwi_dirs']),name='infosource')
infosource.iterables = ('dwi_dirs', dwiDirectories)

# Data grabber DWI
datasource_dwi = pe.Node(interface=nio.DataGrabber(infields=['dwi_dirs'], outfields=info.keys()),
                     name = 'datasource')
datasource_dwi.inputs.base_directory = dataDirectory
datasource_dwi.inputs.template = '*'
datasource_dwi.inputs.field_template = dict(dwi='%s/*.nii*',
                                        bvals='%s/*bval*',
                                        bvecs='%s/*bvec*'
                                        )
datasource_dwi.inputs.template_args = info
datasource_dwi.inputs.sort_filelist = True

# normalize T1 to mni space rigidly
normalizeT1 = func.create_normalize_t1_pipeline(name='normalize_t1')
normalizeT1.inputs.inputnode.in_file = dataDirectory+t1_file

# realign the dwi series to mni space
realignDWI = func.create_realign_dwi_pipeline(name='realign_dwi')
realignDWI.inputs.inputnode.flip_bvals_x = flip_bvals_x

# tensor fit
tensorFit = func.create_tensorfitting_pipeline2(name='tensor_fitting')

# mrtrix CSD
mrtrixCSD = func.create_mrtrix_csd_pipeline(name='mrtrix_csd')

# mrtrix tractography
mrtrixTracking = func.create_mrtrix_tracking_pipeline(name='mrtrix_tracking')
mrtrixTracking.inputs.inputnode.in_roi_lgn_left = rois_files[0]
mrtrixTracking.inputs.inputnode.in_roi_v1_left = rois_files[1]
mrtrixTracking.inputs.inputnode.in_roi_lgn_right = rois_files[2]
mrtrixTracking.inputs.inputnode.in_roi_v1_right = rois_files[3]
mrtrixTracking.inputs.inputnode.in_roi_exclude_left = rois_files[4]
mrtrixTracking.inputs.inputnode.in_roi_exclude_right = rois_files[5]

# fbc measures
fbcMeasures = func.create_fbc_measures_pipeline(name='fbc_measures')

# save the processed data
datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = outputDirectory

# run
main_workflow = pe.Workflow(name="pipeline_files")
main_workflow.base_dir = dataDirectory
main_workflow_pipe = [
						  (infosource, datasource_dwi, [('dwi_dirs', 'dwi_dirs')])

						  ,(datasource_dwi, realignDWI, [('dwi','inputnode.in_file')])
						  ,(datasource_dwi, realignDWI, [('bvecs', 'inputnode.in_bvec')])
						  ,(datasource_dwi, realignDWI, [('bvals', 'inputnode.in_bval')])
						  ,(normalizeT1, realignDWI, [('outputnode.out_t1_brain', 'inputnode.t1_file')])
						  ,(normalizeT1, realignDWI, [('outputnode.out_matrix_file', 'inputnode.t1_mat')])

						  # output
						  ,(normalizeT1, datasink, [('outputnode.out_t1','t1')])
						  ,(normalizeT1, datasink, [('outputnode.out_t1_brain','t1.@brain')])
						  ,(realignDWI, datasink, [('outputnode.motion_corrected','dwi')])
						  ,(realignDWI, datasink, [('outputnode.out_bvec','dwi.@bvec')])
						  ,(realignDWI, datasink, [('outputnode.out_bval','dwi.@bval')])

						  # tensor fit
						  ,(realignDWI, tensorFit, [('outputnode.motion_corrected','inputnode.in_file')])
						  ,(realignDWI, tensorFit, [('outputnode.out_bvec','inputnode.in_bvec')])
						  ,(realignDWI, tensorFit, [('outputnode.out_bval','inputnode.in_bval')])
						  ,(tensorFit, datasink, [('outputnode.out_fa','fa')])
						  ,(tensorFit, datasink, [('outputnode.out_mask','mask')])

						  # csd
						  ,(realignDWI, mrtrixCSD, [('outputnode.motion_corrected','inputnode.in_file')])
						  ,(realignDWI, mrtrixCSD, [('outputnode.out_bvec','inputnode.in_bvec')])
						  ,(realignDWI, mrtrixCSD, [('outputnode.out_bval','inputnode.in_bval')])
						  ,(tensorFit, mrtrixCSD, [('outputnode.out_fa','inputnode.in_fa')])
						  ,(tensorFit, mrtrixCSD, [('outputnode.out_mask','inputnode.in_mask')])
						  ,(mrtrixCSD, datasink, [('outputnode.out_csd','dwi.@csd')])

						]
# tractography
if perform_tracking:
	main_workflow_pipe = main_workflow_pipe + [(mrtrixCSD, mrtrixTracking, [('outputnode.out_csd','inputnode.in_csd')])
						  ,(tensorFit, mrtrixTracking, [('outputnode.out_mask','inputnode.in_mask')])
						  ,(mrtrixTracking, datasink, [('outputnode.out_tracking_L','trackings.@left')])
						  ,(mrtrixTracking, datasink, [('outputnode.out_tracking_R','trackings.@right')])
						]

# FBC measures
if perform_fbc:
	main_workflow_pipe = main_workflow_pipe + [(mrtrixTracking, fbcMeasures, [('outputnode.out_tracking_L','inputnode.in_tracking_L')])
						  ,(mrtrixTracking, fbcMeasures, [('outputnode.out_tracking_R','inputnode.in_tracking_R')])
						  ,(fbcMeasures, datasink, [('outputnode.out_fbc_L','trackings.@left_fbc')])
						  ,(fbcMeasures, datasink, [('outputnode.out_fbc_R','trackings.@right_fbc')])
						]

main_workflow.connect(main_workflow_pipe)
#main_workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
main_workflow.run()
# main_workflow.write_graph(graph2use='flat')


