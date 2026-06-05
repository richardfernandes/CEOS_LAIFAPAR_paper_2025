import ee
import numpy as np

def add_date(image):
    """Adds a date band to a GEE image using the system:time_start

    Args:
        image (ee.Image): GEE image with a system_time_start property

    Returns:
        image (ee.Image): GEE image with a new band named date

    """
    return ee.Image(image).addBands((ee.Image.constant(image.get('system:time_start')).rename('date')).reproject(image.select('B4').projection()))

def add_lonlat(image): 
    """Adds a longitude and latitude to a GEE image 

    Args:
        image (ee.Image): GEE image

    Returns:
        image (ee.Image): GEE image with a new band named Lon and a new band named Lat

    """
    image = ee.Image(image)
    
    return image.addBands((ee.Image.pixelLonLat()).reproject(image.select('B4').projection()))

def add_WorldCover(image):
    """Adds a band with the ESA WorldCover 2021 land cover map to a GEE HLS image

    The projection is based on the S2 B04 band found in S2 GEE collections
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002
    https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002

    The map is documented here https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200.
    

    Args:
        image (ee.Image): GEE HLSS30 or HLSL30 image

    Returns:
        image (ee.Image): GEE HLSS30 or HLSL30 image with a new band named Map

    """
    image = ee.Image(image)

    projection = image.select('B4').projection()

    return image.addBands(ee.ImageCollection("ESA/WorldCover/v200") \
                            .first() \
                            .reproject(projection))




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


# add s2 geomtery bands scaled by 10000
def add_geometry(image):
  return image.addBands(image.metadata('MEAN_INCIDENCE_ZENITH_ANGLE_B4').multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(['cosVZA'])) \
              .addBands(image.metadata('MEAN_SOLAR_ZENITH_ANGLE').multiply(3.1415).divide(180).cos().multiply(10000).toUint16().rename(['cosSZA'])) \
              .addBands(image.metadata('MEAN_SOLAR_AZIMUTH_ANGLE').subtract(image.metadata('MEAN_INCIDENCE_AZIMUTH_ANGLE_B4')).multiply(3.1415).divide(180).cos().multiply(10000).toInt16().rename(['cosRAA']))

  
def sample_S2(site_fcpath,input_icname,output_prefixname,bucket_name,year_list,spatial_buffer,max_cloud_percentage,property_list,sample_scale=30,max_sites=None,shard_size=1000,calendar_range=[1,2,'m']):
    """Samples a S2 image collection for a feature collection of sites using GEE assets and API returning a csv file

    Uses GEE API to sample bands in each HLS image matching search criteria within each site in a feature collection.
    Output is to a google cloud bucke
    



    Args:
        site_fcpath (ee.FeatureCollections):  path to GEE feature collection for sampling
        input_ic (string): input S2 collection name
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
                    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',max_cloud_percentage)) \
                    .filterDate(start_date,end_date) \
                    .map(add_date) \
                    .map(add_lonlat) \
                    .map(add_WorldCover) \
                    .map(add_geometry) \
                    .filter(ee.Filter.calendarRange(calendar_range[0],calendar_range[1],calendar_range[2])) 

        s2cloudless_ic = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
                                 .filterDate(start_date, end_date) \
                                .filter(ee.Filter.calendarRange(calendar_range[0],calendar_range[1],calendar_range[2])) 

        sr_ic = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(primary= sr_ic,\
                                                                          secondary= s2cloudless_ic,\
                                                                          condition= ee.Filter.equals(leftField='system:index',rightField='system:index')))


        for shard in range(1,num_shards+1):

            samples = ee.FeatureCollection(site_list.slice(shard*0,shard*0+shard_size)\
                                                    .map(lambda site: sample_site(site,sr_ic,sample_scale,property_list)))  \
                                                    .flatten().flatten()

            task=ee.batch.Export.table.toCloudStorage(collection=samples, bucket=bucket_name,fileNamePrefix=output_prefixname+str(year)+'yr'+str(shard))
            task.start()

    return 
                            