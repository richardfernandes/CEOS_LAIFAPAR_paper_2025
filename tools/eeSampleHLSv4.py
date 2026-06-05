import ee
import numpy as np

def add_date(image):
    """Adds a date band to a GEE image using the system:time_start

    Args:
        image (ee.Image): GEE image with a system_time_start property

    Returns:
        image (ee.Image): GEE image with a new band named date

    """
    return ee.Image(image).addBands((ee.Image.constant(image.get('system:time_start')).rename('date')).reproject(image.select('Fmask').projection()))

def add_lonlat(image): 
    """Adds a longitude and latitude to a GEE image 

    Args:
        image (ee.Image): GEE image

    Returns:
        image (ee.Image): GEE image with a new band named Lon and a new band named Lat

    """
    image = ee.Image(image)
    
    return image.addBands((ee.Image.pixelLonLat()).reproject(image.select('Fmask').projection()))

def add_WorldCover(image):
    """Adds a band with the ESA WorldCover 2021 land cover map to a GEE HLS image

    The projection is based on the HLS FMask band found in either of the HLSL30 or HLSS30 GEE collections
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002

    The map is documented here https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200.
    

    Args:
        image (ee.Image): GEE HLSS30 or HLSL30 image

    Returns:
        image (ee.Image): GEE HLSS30 or HLSL30 image with a new band named Map

    """
    image = ee.Image(image)

    projection = image.select('Fmask').projection()

    return image.addBands(ee.ImageCollection("ESA/WorldCover/v200") \
                            .first() \
                            .reproject(projection))

def mask_clear_fmask(image):
    """Masks a GEE HLS image product based on the Fmask band

    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002
    

    Args:
        image (ee.Image): GEE HLSS30 or HLSL30 image

    Returns:
        image (ee.Image): GEE HLSS30 or HLSL30 image with updated mask

    """
    image = ee.Image(image)
    qa = image.select('Fmask')
    mask =  qa.uint8() \
                .bitwiseAnd(1).eq(0) \
                .And(qa.bitwiseAnd(1<<1).eq(0)) \
                .And(qa.bitwiseAnd(1<<2).eq(0)) \
                .And(qa.bitwiseAnd(1<<3).eq(0))

    return image.updateMask(mask)



def copy_feature_properties(source_feat, dest_feat, properties_list):
    """
    Copies a specific list of properties from a source feature to a destination feature.
    
    Args:
        source_feat (ee.Feature): The feature containing the original data.
        dest_feat (ee.Feature): The feature you want to add data to.
        properties_list (list): A Python list of strings representing property names.
        
    Returns:
        ee.Feature: The destination feature with the new properties attached.
    """
    return ee.Feature(dest_feat).copyProperties(source_feat, properties_list)
    

                
    
def sample_site(site,sr_ic,sample_scale,property_list):
    """
    Samples all bands in each image in an image collection for a set of site polygons
    and copies over a list of properties form the site polygon to the samples.
    
    Args:
        site (ee.Feature): The feature defining the region sampled
        sr_ic (ee.ImageCollection): The collection of images to sample
        sample_scale (Float): The spatial scale (m) to sample at
        properties_list (list): A Python list of strings representing property names.
        
    Returns:
        (ee.FeatureCollection of ee.FeatureCollections):  Feature collection where each feature is a feature collection of samples for an image
    """
    site = ee.Feature(site)
    sr_ic = ee.ImageCollection(sr_ic)
    property_list = ee.List(property_list)
    
    projection = ee.Projection("EPSG:5070").atScale(sample_scale)
    
    return sr_ic \
            .filterBounds(site.geometry()) \
            .map(lambda image: image \
                                .sample(region=site.geometry(),\
                                         projection=projection,\
                                         scale = sample_scale,\
                                         dropNulls=True,\
                                         geometries=True)
                                .map(lambda fe: copy_feature_properties(site,fe,property_list))
                )

def flatten_all(site_samples):
    """Flattens a feature collection of feature collections of samples for a for a site 

    

    Args:
        site_samples (ee.FeatureCollecion of feature collections): GEE HLSS30 or HLSL30 image

    Returns:
        (feature collection): a feature collections of all samples for the site

    """
    site_samples = ee.FeatureCollection(site_samples)
    return ee.FeatureCollection(site_samples).flatten().flatten()

  
def sample_HLS(site_fcpath,input_icname,output_prefixname,bucket_name,year_list,spatial_buffer,max_cloud_percentage,property_list,sample_scale=30,max_sites=None,shard_size=1000,calendar_range=[1,2,'m']):
    """Samples a HLS image collection for a feature collection of sites using GEE assets and API returning a csv file

    Uses GEE API to sample bands in each HLS image matching search criteria within each site in a feature collection.
    Output is to a google cloud bucke
    



    Args:
        site_fcpath (ee.FeatureCollections):  path to GEE feature collection for sampling
        input_ic (string): input HLS collection name
        output_prefixname (string) : prefix for output files
        year_list (list of integers): list of years to sample
        spatial_buffer (float): spatial buffer in m
        cloud_percentage (float): maximum HLS granule cloud percentage
        property_list (list of strings): HLS image properties to copy to each sample feature
        sample_scale (float): scale in m for sampling
        max_sites (integer): maximum number of sites to sample
        shard_size (integer): maximum number of sites in a output csv file
        bucket_name (string): Google Cloud Platfor bucket name 
        calendar_range (list): paameters for GEE calendarRange filter


    Returns:
        set of CSV files or text to console

    """
 
    site_fcpath = ee.String(site_fcpath)
    input_icname = ee.String(input_icname)
    property_list = ee.List(property_list)
    site_fc = ee.FeatureCollection(site_fcpath)
    site_fc  = ee.FeatureCollection(site_fc.aggregate_array('Plot').distinct() \
                                                            .map(lambda plot: site_fc.filter(ee.Filter.eq('Plot', plot)).first())) \
                                                            .map(lambda fe: fe.set('system:time_start',ee.Date.parse('DD/MM/YYYY',fe.get('Date')))) \
                                                            .map(lambda fe: fe.buffer(distance=spatial_buffer,maxError=1)) \
                                                            
    if max_sites is not None:
        site_fc = site_fc.limit(max_sites)
    
    num_sites = site_fc.size().getInfo()
    num_shards = round( (num_sites/ shard_size) + 0.5)
    site_list = site_fc.toList(num_sites)
    for year in year_list:
        start_date = str(year) + '-01-01'
        end_date = str(year) + '-12-31'
        sr_ic = ee.ImageCollection(input_icname) \
                    .filter(ee.Filter.lte('CLOUD_COVERAGE',max_cloud_percentage)) \
                    .filterDate(start_date,end_date) \
                    .map(add_date) \
                    .map(add_lonlat) \
                    .map(add_WorldCover) \
                    .filter(ee.Filter.calendarRange(calendar_range[0],calendar_range[1],calendar_range[2])) 
        for shard in range(0,num_shards):

            samples = ee.FeatureCollection(site_list.slice(shard*0,shard*0+shard_size)\
                                                    .map(lambda site: sample_site(site,sr_ic,sample_scale,property_list)))  \
                                                    .flatten().flatten()

            task=ee.batch.Export.table.toCloudStorage(collection=samples, bucket=bucket_name,fileNamePrefix=output_prefixname+str(year)+'yr'+str(shard))
            task.start()

    return 
                            