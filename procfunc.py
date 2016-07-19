
'''
    Processing pipeline for DW-MRI
    Author: Stephan Meesters
    2016
'''

import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix
import nipype.algorithms.misc as misc
import os
from nipype.utils.filemanip import split_filename

# Get FSLDIR
fsldir = os.environ["FSLDIR"]

## Wrap SpuriousFibers in command line

from nipype.interfaces.base import (
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    File,
    traits,
    isdefined
)

class SpuriousFibersInputSpec(CommandLineInputSpec):
    in_file = File(desc="TCK file", exists=True, mandatory=True, argstr="--fibers %s",position=0)
    out_file = File(desc="VTK file", argstr="--output %s", name_template="%s_fbc.vtk", name_source="in_file", output_name="fbcout",position=1)
    D33 = traits.Float(argstr='--d33 %s', desc="D33", position=2)
    D44 = traits.Float(argstr='--d44 %s', desc="D44", position=3)
    time = traits.Float(argstr='--t %s', desc="time", position=4)
    subsamplestep = traits.Float(argstr='--subsample-step %s', position=5)
    numanterior = traits.Int(argstr='--num-anterior %s', position=6)
    mindist = traits.Float(argstr='--min-dist %s', position=7)
    maxlength = traits.Float(argstr='--max-length %s', position=8)
    minlength = traits.Float(argstr='--min-length %s', position=9)
    #applybothdirs = traits.Bool(argstr='--apply-both-dirs %s', position=10)

class SpuriousFibersOutputSpec(TraitedSpec):
    out_file = File(desc = "VTK file", exists = True)

class SpuriousFibersTask(CommandLine):
    input_spec = SpuriousFibersInputSpec
    output_spec = SpuriousFibersOutputSpec
    cmd = 'SpuriousFibers'

    def _list_outputs(self):
            outputs = self.output_spec().get()
            outputs['out_file'] = os.path.abspath(self._gen_outfilename())
            return outputs

    def _gen_outfilename(self):
            _, name , _ = split_filename(self.inputs.in_file)
            if isdefined(self.inputs.out_file):
                outname = self.inputs.out_file
            else:
                outname = name + '_fbc.vtk'
            return outname


## Pipeline funcs

def create_normalize_t1_pipeline(name='normalize_t1'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file']),name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_t1', 'out_t1_brain', 'out_matrix_file']),name='outputnode')

    # BET
    bet = pe.Node(fsl.BET(robust=True), name='t1_brain')

    # Coregister the extracted brain with MNI 2mm brain
    normalization = pe.Node(fsl.FLIRT(
                                reference=fsldir+'/data/standard/MNI152_T1_2mm_brain.nii.gz', 
                                interp='spline',
                                cost='corratio', 
                                dof=6,
                                datatype='short',
                                out_file='T1_brain.nii.gz'), 
                            name='normalization')

    # Shadow registration of the original T1 image to MNI
    shadowreg = pe.Node(fsl.FLIRT(
                                reference=fsldir+'/data/standard/MNI152_T1_1mm_brain.nii.gz', 
                                interp='spline',
                                apply_xfm=True,
                                datatype='short',
                                out_file='T1.nii.gz'), 
                            name='shadowreg')

    # create and connect pipeline
    pipeline = pe.Workflow(name=name)
    pipeline.connect([
                      (inputnode, bet, [('in_file', 'in_file')]),
		      #(bet, normalization, [('out_file', 'reference')]), # temp
                      (bet, normalization, [('out_file', 'in_file')]),
		      #(bet, shadowreg, [('out_file', 'reference')]), # temp
                      (inputnode, shadowreg, [('in_file', 'in_file')]),
                      (normalization, shadowreg, [('out_matrix_file', 'in_matrix_file')]),

                      # output
                      (normalization, outputnode, [('out_file', 'out_t1_brain')]),
                      (shadowreg, outputnode, [('out_file', 'out_t1')]),
                      (normalization, outputnode, [('out_matrix_file', 'out_matrix_file')])
                    ])

    return pipeline


def create_realign_dwi_pipeline(name='realign_dwi'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'in_bvec', 'in_bval', 't1_file', 't1_mat', 'flip_bvals_x']),
                        name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['motion_corrected', 'out_bvec', 'out_bval']),
                        name='outputnode')

    # split the 4d volume into 3d volumes
    split = pe.Node(fsl.Split(dimension='t'), name='split')

    # select the first image of the volume series as the b=0 image
    pick_ref = pe.Node(niu.Select(index=0), name='pick_ref')

    # align the b=0 image to the t1 image
    normalization = pe.Node(fsl.FLIRT( 
                                interp='trilinear',
                                cost='corratio', 
                                dof=6,
                                datatype='short'), 
                            name='normalization')

    # coregister all individual volumes to the b=0 volume with iteration
    coregistration = pe.MapNode(fsl.FLIRT(
                                no_search=True, 
                                interp='trilinear',
                                dof=6,
                                datatype='short'), name='coregistration', iterfield=['in_file'])

    # transform the bvecs
    rotate_bvecs = pe.Node(niu.Function(input_names=['in_bvec', 'in_matrix', 'flip_bvals_x'], 
                                        output_names=['out_file'], 
                                        function=_rotate_bvecs), 
                            name='rotate_b_matrix')

    # merge the realigned images into a 4D volume again
    merge = pe.Node(fsl.Merge(dimension='t',merged_file='DWI_aligned.nii.gz'), name='merge')

    # rename bvals
    rename = pe.Node(niu.Rename(format_string="DWI_aligned",
                                         keep_ext=True),
                             name="rename_bvals")

    pipeline = pe.Workflow(name=name)
    pipeline.connect([
                       (inputnode, split, [('in_file', 'in_file')])
                      ,(split, pick_ref, [('out_files', 'inlist')])

                      ,(pick_ref, normalization, [('out', 'in_file')])
                      ,(inputnode, normalization, [('t1_file', 'reference')])
                      ,(inputnode, normalization, [('t1_mat', 'in_matrix_file')])

                      ,(split, coregistration, [('out_files', 'in_file')])
                      ,(normalization, coregistration, [('out_file', 'reference')])
                      ,(normalization, coregistration, [('out_matrix_file', 'in_matrix_file')])
                      ,(coregistration, merge, [('out_file', 'in_files')])

                      ,(inputnode, rotate_bvecs, [('in_bvec', 'in_bvec')])
                      ,(inputnode, rotate_bvecs, [('flip_bvals_x', 'flip_bvals_x')])
                      ,(coregistration, rotate_bvecs, [('out_matrix_file', 'in_matrix')])

                      ,(merge, outputnode, [('merged_file', 'motion_corrected')])
                      ,(rotate_bvecs, outputnode, [('out_file', 'out_bvec')])

                      ,(inputnode, rename, [('in_bval','in_file')])
                      ,(rename, outputnode, [('out_file','out_bval')])
                    ])

    return pipeline



def _rotate_bvecs(in_bvec, in_matrix, flip_bvals_x):
    import os
    import numpy as np

    name, fext = os.path.splitext(os.path.basename(in_bvec))
    if fext == '.gz':
        name, _ = os.path.splitext(name)
    out_file = os.path.abspath('./DWI_aligned.bvec')
    bvecs = np.loadtxt(in_bvec)
    new_bvecs = np.zeros(shape=bvecs.T.shape) #pre-initialise array, 3 col format

    for i, vol_matrix in enumerate(in_matrix[0::]): #start index at 0
        bvec = np.matrix(bvecs[:, i])
        rot = np.matrix(np.loadtxt(vol_matrix)[0:3, 0:3])
        new_bvecs[i] = (np.array(rot * bvec.T).T)[0] #fill each volume with x,y,z as we go along
        if flip_bvals_x:
          new_bvecs[i][0] = -new_bvecs[i][0] # flip bval x coordinate
    np.savetxt(out_file, np.array(new_bvecs).T, fmt='%0.15f')
    return out_file

    

def create_tensorfitting_pipeline(name='tensor_fitting'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'in_bvec', 'in_bval']),
                        name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_tensors','out_fa','out_v1','out_mask']),
                        name='outputnode')

    # get b=0 image
    fslroi = pe.Node(interface=fsl.ExtractROI(t_min=0,t_size=1),name='fslroi')

    # create brain mask from b=0 image
    bet = pe.Node(interface=fsl.BET(mask=True,frac=0.34,out_file='DWI_aligned.nii.gz'),name='bet')

    # fit tensors
    dtifit = pe.Node(interface=fsl.DTIFit(base_name='DWI_aligned',save_tensor = True),name='dtifit')

    pipeline = pe.Workflow(name)
    pipeline.connect([
                         (inputnode, fslroi, [('in_file', 'in_file')])
                        ,(fslroi,bet,[('roi_file','in_file')])
                        ,(bet,dtifit,[('mask_file','mask')])

                        ,(inputnode, dtifit, [('in_file', 'dwi')])
                        ,(inputnode, dtifit, [('in_bval', 'bvals')])
                        ,(inputnode, dtifit, [('in_bvec', 'bvecs')])

                        ,(bet, outputnode, [('mask_file','out_mask')])
                        ,(dtifit, outputnode, [('tensor','out_tensors')])
                        ,(dtifit, outputnode, [('FA','out_fa')])
                        ,(dtifit, outputnode, [('V1','out_v1')])

                      ])

    return pipeline



def create_tensorfitting_pipeline2(name='tensor_fitting'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'in_bvec', 'in_bval']),
                        name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_tensors','out_fa','out_v1','out_mask']),
                        name='outputnode')

    # get b=0 image
    fslroi = pe.Node(interface=fsl.ExtractROI(t_min=0,t_size=1),name='fslroi')

    # create brain mask from b=0 image
    bet = pe.Node(interface=fsl.BET(mask=True,frac=0.34,out_file='DWI_aligned.nii.gz'),name='bet')


    # convert bvals and bvecs into mrtrix format
    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(),name='fsl2mrtrix')

    # fit tensors
    mrtrixdtifit = pe.Node(interface=mrtrix.DWI2Tensor(),name='mrtrixdtifit')	

    # output major eigenvectors
    mrtrixv1 = pe.Node(interface=mrtrix.Tensor2Vector(),name='mrtrixv1')

    # output FA
    mrtrixFA = pe.Node(interface=mrtrix.Tensor2FractionalAnisotropy(),name='mrtrixFA')

    pipeline = pe.Workflow(name)
    pipeline.connect([
                         (inputnode, fslroi, [('in_file', 'in_file')])
                        ,(fslroi,bet,[('roi_file','in_file')])

                        ,(inputnode, fsl2mrtrix, [('in_bvec', 'bvec_file')])
                        ,(inputnode, fsl2mrtrix, [('in_bval', 'bval_file')])

                        ,(inputnode, mrtrixdtifit, [('in_file', 'in_file')])
                        ,(fsl2mrtrix, mrtrixdtifit, [('encoding_file', 'encoding_file')])

                        ,(mrtrixdtifit, mrtrixv1, [('tensor','in_file')])
                        ,(mrtrixdtifit, mrtrixFA, [('tensor','in_file')])


                        ,(bet, outputnode, [('mask_file','out_mask')])
                        ,(mrtrixdtifit, outputnode, [('tensor','out_tensors')])
                        ,(mrtrixv1, outputnode, [('vector','out_v1')])
                        ,(mrtrixFA, outputnode, [('FA','out_fa')])

                      ])

    return pipeline

def create_mrtrix_csd_pipeline(name='mrtrix_csd'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_file', 'in_bvec', 'in_bval', 'in_mask', 'in_fa']),
                        name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_csd']),
                        name='outputnode')

    
    # convert bvals and bvecs into mrtrix format
    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(),name='fsl2mrtrix')

    gunzip = pe.Node(interface=misc.Gunzip(), name='gunzip')

    # create eroded mask
    erodedmask = pe.Node(interface=mrtrix.Erode(number_of_passes=3),name='erodedmask')

    # create multiplied mask
    multiplmask = pe.Node(interface=mrtrix.MRMultiply2(),name='multiplmask')

    # create single voxel mask
    single_voxel_mask = pe.Node(interface=mrtrix.Threshold(absolute_threshold_value = 0.7),name='single_voxel_mask')

    estimateresponse = pe.Node(interface=mrtrix.EstimateResponseForSH(),name='estimateresponse')
    #estimateresponse.inputs.maximum_harmonic_order = 8   # handled automatically by mrtrix

    csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),name='csdeconv')
    #csdeconv.inputs.maximum_harmonic_order = 8		  # handled automatically by mrtrix


    pipeline = pe.Workflow(name)
    pipeline.connect([
                         (inputnode, fsl2mrtrix, [('in_bvec', 'bvec_file')])
                        ,(inputnode, fsl2mrtrix, [('in_bval', 'bval_file')])
                        ,(inputnode, gunzip,[("in_file","in_file")])

                        ,(inputnode, erodedmask, [("in_mask","in_file")])
                        ,(erodedmask,multiplmask,[("out_file","in_file_1")])
                        ,(inputnode,multiplmask,[("in_fa","in_file_2")])
                        ,(multiplmask,single_voxel_mask,[("out_file","in_file")])

                        ,(gunzip, estimateresponse,[("out_file","in_file")])
                        ,(fsl2mrtrix, estimateresponse,[("encoding_file","encoding_file")])
                        #,(inputnode, estimateresponse,[("in_mask","mask_image")])
                        ,(single_voxel_mask, estimateresponse,[("out_file","mask_image")])

                        

                        ,(gunzip, csdeconv,[("out_file","in_file")])
                        ,(inputnode, csdeconv,[("in_mask","mask_image")])
                        ,(estimateresponse, csdeconv,[("response","response_file")])
                        ,(fsl2mrtrix, csdeconv,[("encoding_file","encoding_file")])

                        ,(csdeconv, outputnode,[("spherical_harmonics_image","out_csd")])
                      ])

    return pipeline





def create_mrtrix_tracking_pipeline(name='mrtrix_tracking'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_csd','in_mask','in_roi_lgn_left','in_roi_v1_left','in_roi_lgn_right','in_roi_v1_right','in_roi_exclude_left','in_roi_exclude_right']),
                        name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_tracking_L','out_tracking_R']),
                        name='outputnode')


    #nodes
    probCSDstreamtrack_L = pe.Node(interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(inputmodel = 'SD_PROB', desired_number_of_tracks = 20000,unidirectional = True, maximum_number_of_tracks = 0, out_file='probCSDLnew_stop.tck'),name='probCSDstreamtrack_L', stop = True, maximum_tract_length=114)

    probCSDstreamtrack_R = pe.Node(interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(inputmodel = 'SD_PROB', desired_number_of_tracks = 1,unidirectional = True, maximum_number_of_tracks = 0, out_file="probCSDRnew_stop.tck"),name='probCSDstreamtrack_R', stop = True, maximum_tract_length=114)

    probCSDstreamtrack_L.iterables = ("out_file", ["probCSDLnew_stop"+str(x)+".tck" for x in range(1,11)])
    probCSDstreamtrack_R.iterables = ("out_file", ["probCSDRnew_stop"+str(x)+".tck" for x in range(1,2)])
    
    pipeline = pe.Workflow(name)
    pipeline.connect([
                        (inputnode, probCSDstreamtrack_L, [("in_csd", "in_file")])
                        ,(inputnode, probCSDstreamtrack_L, [("in_mask", "mask_file")])
                        ,(inputnode, probCSDstreamtrack_L,[("in_roi_lgn_left","seed_file")])
                        ,(inputnode, probCSDstreamtrack_L,[("in_roi_v1_left","include_file")])
                        ,(inputnode, probCSDstreamtrack_L,[("in_roi_exclude_right","exclude_file")])

                        ,(inputnode, probCSDstreamtrack_R, [("in_csd", "in_file")])
                        ,(inputnode, probCSDstreamtrack_R, [("in_mask", "mask_file")])
                        ,(inputnode, probCSDstreamtrack_R,[("in_roi_lgn_right","seed_file")])
                        ,(inputnode, probCSDstreamtrack_R,[("in_roi_v1_right","include_file")])
                        ,(inputnode, probCSDstreamtrack_R,[("in_roi_exclude_left","exclude_file")])

                        ,(probCSDstreamtrack_L, outputnode,[("tracked","out_tracking_L")])
                        ,(probCSDstreamtrack_R, outputnode,[("tracked","out_tracking_R")])
                      ])

    return pipeline


def create_fbc_measures_pipeline(name='fbc_measures'):

    # inputs
    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['in_tracking_L','in_tracking_R']),name='inputnode')

    # outputs
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['out_fbc_L','out_fbc_R']),
                        name='outputnode')

    # nodes
    fbc_L = pe.Node(interface=SpuriousFibersTask(D33=1.0, D44=0.04, time=1.4, numanterior=1000), name='fbc_L', stop=True)
    fbc_R = pe.Node(interface=SpuriousFibersTask(D33=1.0, D44=0.04, time=1.4, numanterior=1000), name='fbc_R', stop=True)

    # pipeline
    pipeline = pe.Workflow(name)
    pipeline.connect([
                        (inputnode, fbc_L, [("in_tracking_L", "in_file")])
                        ,(inputnode, fbc_R, [("in_tracking_R", "in_file")])
                        ,(fbc_L, outputnode,[("out_file","out_fbc_L")])
                        ,(fbc_R, outputnode,[("out_file","out_fbc_R")])
                      ])

    return pipeline

