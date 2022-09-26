//
// Created by cheng on 19.09.22.
//

#pragma once


#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>


#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/loss_function.h>


#include <iostream>
#include <vector>

// change functions here to a brdf class

namespace DSONL {

	#define RECIPROCAL_PI 0.3183098861837907
	using namespace std;
	using namespace cv;
	//helper functions
	float lerp_1d(float fst, float sec, float by){return fst*(1-by)+ sec*by;}
	Vec3f lerp_3d( Vec3f firstVector, Vec3f secondVector, float by){

		float retX = lerp_1d(firstVector.val[0], secondVector.val[0], by);
		float retY = lerp_1d(firstVector.val[1], secondVector.val[1], by);
		float retZ = lerp_1d(firstVector.val[2], secondVector.val[2], by);
		Vec3f u(retX, retY, retZ);
		return  u;
	}
	template <class T> T clamp(T x, T min, T max){if (x>max){ return max;}if (x<min){return  min;}return x;}
	float SchlickFresnel(float i){
		float x = clamp(1.0-i, 0.0, 1.0);
		float x2 = x*x;
		return x2*x2*x;
	}


	class BrdfMicrofacet {
	private:
		Vec3f L; // light direction in world space
		Vec3f cameraPosition; // camera position in world space
		Vec3f N; // normal
		Vec3f H; // halfDirection
		Vec3f V;
		Vec3f baseColor;
		float IOR=0.04; //Fresnel IOR (index of refraction), _Ior("Ior",  Range(0,4)) = 1.5
		float metallicValue;
		float roughnessValue; //  roughness = 1 - _Glossiness;??????????
		float LdotH, NdotH, NdotL, NdotV;
		Vec3f baseColorTexture; // base color
		//helper functions
		float _squared(float x){return x*x;}
		float _dot(Vec3f&  fst, Vec3f&  snd){ return max( (float)0.0, fst.dot(snd));}
	public:
		BrdfMicrofacet(const Vec3f&  L_, const Vec3f&  N_, const Vec3f& view_beta,
					   const float& roughnessValue_,
					   const float& metallicValue_,
					   const Vec3f& baseColor_ // RGB order
					   ){
			L=L_;
			V=view_beta; float shiftAmount = N_.dot(view_beta);
			N= shiftAmount < 0.0f ? N_ + view_beta * (-shiftAmount + 1e-5f) : N_; //normal direction calculations
			H= normalize(view_beta+L_);
			LdotH= _dot(L,H);
			NdotH= _dot(N,H);
			NdotL= _dot(N,L);
			NdotV= _dot(N,V);
			baseColor=baseColor_;
			roughnessValue=roughnessValue_;
			metallicValue=metallicValue_;
			D= GGXNormalDistribution(roughnessValue, NdotH);// calculate the normal distribution function result
            F=NewSchlickFresnelFunction(IOR,baseColor, LdotH,metallicValue);// calculate the Fresnel reflectance
			G=AshikhminShirleyGeometricShadowingFunction( NdotL,  NdotV,  LdotH);
			specularityVec=specularity(F,D,G, NdotL, NdotV);
			diffuseColorVec=diffuseColor(baseColor, metallicValue);
			lightingModel= brdfMicrofacet(baseColor,metallicValue, specularityVec, NdotL);
			radiance=0.299*lightingModel.val[0]+0.587*lightingModel.val[1]+0.114*lightingModel.val[2];// 0R,1G,2B
		}
        float D;
		Vec3f F;
		float G;
		Vec3f specularityVec;
		Vec3f diffuseColorVec;
		Vec3f lightingModel;
		float radiance;

		float GGXNormalDistribution(float roughness, float NdotH);
		Vec3f NewSchlickFresnelFunction(float ior, Vec3f Color, float LdotH, float Metallicness);
		float AshikhminShirleyGeometricShadowingFunction (float NdotL, float NdotV, float LdotH);
		Vec3f specularity( Vec3f FresnelFunction,Vec3f SpecularDistribution, float GeometricShadow, float NdotL,float NdotV );
		Vec3f diffuseColor ( Vec3f baseColor ,float _Metallic );
		Vec3f brdfMicrofacet (  Vec3f Color_rgb ,float _Metallic, Vec3f specularity, float  NdotL);

	};

	float BrdfMicrofacet::GGXNormalDistribution(float roughness, float NdotH)
	{

		float roughnessSqr = roughness*roughness;

		float NdotHSqr = NdotH*NdotH;

		float TanNdotHSqr = (1-NdotHSqr)/NdotHSqr;

		return (1.0/3.1415926535) * _squared(roughness/(NdotHSqr * (roughnessSqr + TanNdotHSqr)));

		// float denom = NdotHSqr * (roughnessSqr-1)

	}

	Vec3f BrdfMicrofacet:: NewSchlickFresnelFunction(float ior, Vec3f Color, float LdotH, float Metallicness){

		Vec3f f0 = Vec3f(0.16*ior*ior,0.16*ior*ior,0.16*ior*ior);
		Vec3f F0 = lerp_3d(f0,Color,Metallicness);
		Vec3f one(1,1,1);
		return F0 + (one - F0) * SchlickFresnel(LdotH);

	}

	float BrdfMicrofacet:: AshikhminShirleyGeometricShadowingFunction (float NdotL, float NdotV, float LdotH){
		float Gs = NdotL*NdotV/(LdotH*max(NdotL,NdotV));
		return  (Gs);
	}

	Vec3f BrdfMicrofacet::specularity( Vec3f FresnelFunction,Vec3f SpecularDistribution, float GeometricShadow, float NdotL,float NdotV ){

		return (SpecularDistribution.mul(FresnelFunction)  * GeometricShadow) / (4 * (  NdotL * NdotV));
	}

    //  Color_rgb is the texture base color
	Vec3f BrdfMicrofacet:: diffuseColor ( Vec3f baseColor ,float _Metallic ){

		return baseColor * (1.0 - _Metallic)*RECIPROCAL_PI;
	}

	Vec3f BrdfMicrofacet:: brdfMicrofacet (  Vec3f baseColor ,float _Metallic, Vec3f specularity, float  NdotL){

		Vec3f diffuseColor_=diffuseColor( baseColor ,_Metallic);

		Vec3f lightingModel = specularity+diffuseColor_;

		lightingModel *= NdotL;

		return lightingModel;
	}


	void runEM(){

	}

	void updateDelta(
			         const Sophus::SE3d CurrentT,
					 const Eigen::Matrix3d & K,
					 const Mat& image_baseColor,
					 const Mat& depth_map,
					 const Mat& image_metallic,
					 const Mat& image_roughnes,
					 const Eigen::Vector3d & L2c1,
					 Mat& deltaMap
					 ){
		double fx = K(0, 0), cx = K(0, 2), fy =  K(1, 1), cy = K(1, 2);
		for (int u = 0; u< depth_map.rows; u++) // colId, cols: 0 to 480
		{
			for (int v = 0; v < depth_map.cols; v++) // rowId,  rows: 0 to 640
			{
				if (depth_map.at<double>(u,v) < 1e-3 ) { continue; }
				Eigen::Vector2d pixelCoord((double)v,(double)u);//  u is the row id , v is col id
				double d=depth_map.at<double>(u,v);
				double d_x1= depth_map.at<double>(u,v+1);
				double  d_y1= depth_map.at<double>(u+1, v);
				// calculate 3D point coordinate
				Eigen::Vector3d p_3d_no_d((pixelCoord(0)-cx)/fx, (pixelCoord(1)-cy)/fy,1.0);
				Eigen::Vector3d p_c1=d*p_3d_no_d;


				// calculate normal for each point
				Eigen::Matrix<double, 3,1> normal, v_x, v_y;
				v_x <<  ((d_x1-d)*(v-cx)+d_x1)/fx, (d_x1-d)*(u-cy)/fy , (d_x1-d);
				v_y << (d_y1-d)*(v-cx)/fx,(d_y1+ (d_y1-d)*(u-cy))/fy, (d_y1-d);
				v_x=v_x.normalized();
				v_y=v_y.normalized();
				normal=v_x.cross(v_y);
				normal=normal.normalized();

				// calculate alpha_1
				Eigen::Matrix<double,3,1> alpha_1= L2c1-p_c1; alpha_1=alpha_1.normalized();
				// calculate beta and beta_prime;
				Eigen::Matrix<double,3,1> beta,beta_prime;
				beta=-p_c1;
				beta=beta.normalized();
				beta_prime=-CurrentT.rotationMatrix().transpose()*CurrentT.translation()-p_c1;
				beta_prime=beta_prime.normalized();

				// baseColor_bgr[3],metallic, roughness
				float metallic=image_metallic.at<float>(u,v);
				float roughness =image_roughnes.at<float>(u,v);
				//Instantiation of BRDF object
				Vec3f baseColor(image_baseColor.at<Vec3b>(u,v)[2],image_baseColor.at<Vec3b>(u,v)[1],image_baseColor.at<Vec3b>(u,v)[0]);
				Vec3f L_(alpha_1(0),alpha_1(1),alpha_1(2));
				Vec3f N_(normal(0),normal(1),normal(2));
				Vec3f View_beta(beta(0),beta(1),beta(2));
				Vec3f View_beta_prime(beta_prime(0),beta_prime(1),beta_prime(2));

				BrdfMicrofacet radiance_beta_vec(L_,N_,View_beta,(float )roughness,(float)metallic,baseColor);
				double radiance_beta= radiance_beta_vec.radiance;
				BrdfMicrofacet radiance_beta_prime_vec(L_,N_,View_beta_prime,(float )roughness,(float)metallic,baseColor);
				double radiance_beta_prime= radiance_beta_prime_vec.radiance;
				double delta= radiance_beta/radiance_beta_prime;
				// store delta in the delta map
				deltaMap.at<double>(u,v)= delta;


			}
		}









	}














}