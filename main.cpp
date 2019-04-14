#include <iostream>
#include <memory>
#include <vector>
using namespace std;

#include <opencv2/opencv.hpp>

//using namespace cv;

class LandMark
{
public:
    bool triangulate_;
    cv::Point3f point_;
    std::list<int> view_id_;
    std::list<cv::Point2f> view_pt_;
};

class Pose
{
public:
    Pose() {}
    Pose(cv::Mat K, cv::Mat R, cv::Mat T) :K_(K), R_(R), T_(T) {}
    cv::Mat K_;
    cv::Mat R_;
    cv::Mat T_;

    cv::Mat GetP()
    {
        cv::Mat P = cv::Mat(3, 4, CV_64FC1);
        P.at<double>(0, 0) = R_.at<double>(0, 0); P.at<double>(0, 1) = R_.at<double>(0, 1); P.at<double>(0, 2) = R_.at<double>(0, 2); P.at<double>(0, 3) = T_.at<double>(0, 0);
        P.at<double>(1, 0) = R_.at<double>(1, 0); P.at<double>(1, 1) = R_.at<double>(1, 1); P.at<double>(1, 2) = R_.at<double>(1, 2); P.at<double>(1, 3) = T_.at<double>(1, 0);
        P.at<double>(2, 0) = R_.at<double>(2, 0); P.at<double>(2, 1) = R_.at<double>(2, 1); P.at<double>(2, 2) = R_.at<double>(2, 2); P.at<double>(2, 3) = T_.at<double>(2, 0);

        return K_*P;
    }
};

class Map
{
public:
    bool INIT_;
    int KEY_FRAME_NUMBER_;

    std::list<int>     key_frames_id_;
    std::map<int, Pose> key_frames_poses_;
    std::list<LandMark*> landmarks_;

    int last_key_frame_id_;
    cv::Mat last_key_frame_;
    std::vector<LandMark*> last_key_frame_landmarks_;
    std::vector<cv::Point2f> last_key_frame_points;

    cv::Mat last_frame_;
    std::vector<cv::Mat>     last_frame_pyramid;
    std::vector<cv::Point2f> last_frame_points;
    std::vector<int>         last_frame_tracked_last_key_frame_id_;

    void OptimiseMap()
    {

    }
    void UpdateMap()
    {
        // Delete 
    }
    void SaveMap()
    {

    }
};


int main(int argc, const char* argv[])
{

    // Input
    std::string video_path = argv[1];

    // Camera Parameter
    cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1);
    K.at<double>(0, 0) = 2437.500691842076;
    K.at<double>(1, 1) = 2437.500691842076;
    K.at<double>(0, 2) = 1917.8327084486385;
    K.at<double>(1, 2) = 1078.9128993926025;
    K.at<double>(2, 2) = 1;
    cv::Mat distort = cv::Mat::zeros(1, 5, CV_64FC1);
    distort.at<double>(0, 0) = 0.008639906356385286;
    distort.at<double>(0, 1) = -0.016480669822877878;
    distort.at<double>(0, 2) = 0;
    distort.at<double>(0, 3) = 0;
    distort.at<double>(0, 4) = 0.013240310781908701;

    // Corner
    int POINT_DISTANCE = 50;
    int MAX_POINT = 1000;
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 40, 0.01);
    cv::Size subPixWinSize(10, 10), winSize(31, 31);

    // Image
    int down_scale = 1;
    int image_width = 3840.0;
    int image_height = 2160.0;
    int border = std::min(image_width, image_height)*0.1;

    // Map
    int KEY_FRAME_PIXEL = 100;

    // Border Mask
    cv::Mat mask = cv::Mat(image_height, image_width, CV_8UC1, cv::Scalar(255));
    for (int y = 0; y < image_height; y++)
    {
        for (int x = 0; x < image_width; x++)
        {
            if (x<border || x>(image_width - border) || y<border || y>(image_height - border))
                mask.at<unsigned char>(y, x) = 0;
        }
    }

    // Init a map
    Map mono_vo_map;
    mono_vo_map.INIT_ = false;

    // Open Video
    cv::VideoCapture video(video_path);

    if (!video.isOpened())
    {
        std::cout << "Read video Failed !" << std::endl;
        return -1;
    }
    int frame_num = video.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Total frame number is: " << frame_num << std::endl;

    //video.set(CV_CAP_PROP_POS_FRAMES, 600);

    // Current Pose
    // ×¢ÒâÇ³¿½±´
    cv::Mat cur_r = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat cur_t = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat cur_r_vector;
    cv::Mat cur_frame;
    std::vector<cv::Point2f> cur_points;
    std::vector<cv::Mat> cur_frame_pyramid;

    cv::namedWindow("Video", cv::WINDOW_KEEPRATIO);
    for (int i = 0; i < frame_num; ++i)
    {
        cv::Mat cur_frame_orgin;

        //video >> cur_frame_orgin;
        //undistort(cur_frame_orgin, cur_frame, K, distort, K);

        video >> cur_frame;

        cv::Mat cur_frame_gray;
        cvtColor(cur_frame, cur_frame_gray, CV_BGR2GRAY);

        cv::equalizeHist(cur_frame_gray, cur_frame_gray);

        // Clear Cur Frame
        cur_frame_pyramid.clear();
        cur_points.clear();

        buildOpticalFlowPyramid(cur_frame_gray, cur_frame_pyramid, cv::Size(31, 31), 4);

        if (i == 0)
        {
            // Init Features
            goodFeaturesToTrack(cur_frame_gray, cur_points, MAX_POINT, 0.01, POINT_DISTANCE, mask);
            cornerSubPix(cur_frame_gray, cur_points, subPixWinSize, cv::Size(-1, -1), termcrit);

            // Store to Map
            // Set The First Frame As KeyFrame
            mono_vo_map.key_frames_id_.push_back(i);
            mono_vo_map.key_frames_poses_.insert({ i,Pose(K,cur_r.clone(),cur_t.clone()) });
            // Update Last Key Frame
            mono_vo_map.last_key_frame_id_ = i;
            mono_vo_map.last_key_frame_ = cur_frame.clone();
            mono_vo_map.last_key_frame_points.clear();
            mono_vo_map.last_key_frame_landmarks_.clear();
            for (int j = 0; j < cur_points.size(); j++)
            {
                LandMark *landmark = new LandMark();
                landmark->triangulate_ = false;
                landmark->view_id_.push_back(i);
                landmark->view_pt_.push_back(cur_points[j]);

                // Add to Last Key Frame
                mono_vo_map.last_key_frame_landmarks_.push_back(landmark);
                mono_vo_map.last_key_frame_points.push_back(cur_points[j]);

                // Add Landmarks
                mono_vo_map.landmarks_.push_back(landmark);
            }

            // Update Last Frame
            mono_vo_map.last_frame_ = cur_frame.clone();
            mono_vo_map.last_frame_points.clear();
            mono_vo_map.last_frame_pyramid.clear();
            mono_vo_map.last_frame_tracked_last_key_frame_id_.clear();

            mono_vo_map.last_frame_pyramid.assign(cur_frame_pyramid.begin(), cur_frame_pyramid.end());
            for (int j = 0; j < cur_points.size(); j++)
            {
                mono_vo_map.last_frame_points.push_back(cur_points[j]);
                mono_vo_map.last_frame_tracked_last_key_frame_id_.push_back(j);
            }
        }
        else
        {
            // Track With The Last Frame
            std::vector<uchar> track_status;
            std::vector<float> track_err;
            calcOpticalFlowPyrLK(mono_vo_map.last_frame_pyramid, cur_frame_pyramid,
                mono_vo_map.last_frame_points, cur_points,
                track_status, track_err,
                cv::Size(31, 31), 4);

            // Update track_status by border mask
            for (int j = 0; j < track_status.size(); j++)
            {
                if (track_status[j])
                {
                    cv::Point2f &point = cur_points[j];
                    if (point.x < border || point.x>(image_width - border) || point.y<border || point.y >(image_height - border))
                    {
                        track_status[j] = 0;
                    }
                }
            }

            // Track with The last key frame
            // Check Key Frame
            int tracked_point_count = 0;
            double delta_pixel = 0;
            for (int j = 0; j < track_status.size(); j++)
            {
                if (track_status[j])
                {
                    int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[j];
                    delta_pixel += std::max(std::abs(cur_points[j].x - mono_vo_map.last_key_frame_points[last_key_frame_point_id].x), std::abs(cur_points[j].y - mono_vo_map.last_key_frame_points[last_key_frame_point_id].y));
                    tracked_point_count++;
                }
            }
            delta_pixel /= tracked_point_count;


            if (delta_pixel > KEY_FRAME_PIXEL)
            {
                // Found A Key Frame
                //std::cout << "Found a key frame...." << std::endl;

                // Match By L-K
                std::vector<cv::Point2f> last_key_frame_points_tracked, cur_points_tracked;
                std::vector<int> tracked_id;

                // Already Triangulate Points
                std::vector<cv::Point2f> cur_points_tracked_already_triangulate;
                std::vector<int> already_triangulate_id;

                // Match After RANSAC
                std::vector<cv::Point2f> last_key_frame_points_tracked_filter, cur_points_tracked_filter;
                std::vector<int> tracked_filter_id;
                std::vector<LandMark*> tracked_filter_landmark;

                // Needed Triangulate Points
                std::vector<cv::Point2f> cur_points_tracked_needed_triangulate;
                std::vector<int> needed_triangulate_id;


                // Get Match After L-K
                for (int j = 0; j < track_status.size(); j++)
                {
                    if (track_status[j])
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[j];
                        last_key_frame_points_tracked.push_back(mono_vo_map.last_key_frame_points[last_key_frame_point_id]);
                        cur_points_tracked.push_back(cur_points[j]);
                        tracked_id.push_back(j);
                    }
                }


                // Init
                if (!mono_vo_map.INIT_)
                {
                    // Set Init True
                    mono_vo_map.INIT_ = true;

                    // RANSAC With Essential
                    vector<uchar> ransac_filter;
                    cv::Mat essential_mat = findEssentialMat(last_key_frame_points_tracked, cur_points_tracked, K, cv::RANSAC, 0.99899, 3.0, ransac_filter);

                    for (int j = 0; j < tracked_id.size(); j++)
                    {
                        if (ransac_filter[j])
                        {
                            int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[tracked_id[j]];

                            last_key_frame_points_tracked_filter.push_back(mono_vo_map.last_key_frame_points[last_key_frame_point_id]);
                            cur_points_tracked_filter.push_back(cur_points[tracked_id[j]]);
                            tracked_filter_id.push_back(tracked_id[j]);
                        }
                    }

                    cv::recoverPose(essential_mat, last_key_frame_points_tracked_filter, cur_points_tracked_filter, K, cur_r, cur_t);


                    std::cout << cur_t.at<double>(0, 0) << " " << cur_t.at<double>(1, 0) << " " << cur_t.at<double>(2, 0) << std::endl;

                    // needed_triangulate_id
                    for (int j = 0; j < tracked_filter_id.size(); j++)
                    {
                        cur_points_tracked_needed_triangulate.push_back(cur_points[tracked_filter_id[j]]);
                        needed_triangulate_id.push_back(tracked_filter_id[j]);
                    }
                    // not triangulate 
                    tracked_filter_id.clear();
                    last_key_frame_points_tracked_filter.clear();
                    cur_points_tracked_filter.clear();

                }
                else
                {

                    // Get the Tracked which has triangulated
                    for (int j = 0; j < tracked_id.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[tracked_id[j]];

                        if (mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->triangulate_)
                        {
                            cur_points_tracked_already_triangulate.push_back(cur_points[tracked_id[j]]);
                            already_triangulate_id.push_back(tracked_id[j]);

                            // Delete The filtered landmark
                            // Too bad
                            mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->triangulate_ = false;
                        }
                        else
                        {
                            cur_points_tracked_needed_triangulate.push_back(cur_points[tracked_id[j]]);
                            needed_triangulate_id.push_back(tracked_id[j]);
                        }
                    }


                    // Get Landmark By Track
                    std::vector<cv::Point3d> landmark_3d;
                    for (int j = 0; j < already_triangulate_id.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[already_triangulate_id[j]];

                        landmark_3d.push_back(mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->point_);
                    }


                    // RANSAC With SolvePnP
                    cv::Rodrigues(cur_r, cur_r_vector);
                    vector<int> ransac_filter;
                    solvePnPRansac(landmark_3d, cur_points_tracked_already_triangulate, K, cv::Mat(), cur_r_vector, cur_t, true, 100, 8, 0.9899, ransac_filter, cv::SOLVEPNP_EPNP);


                    std::vector<cv::Point3d> landmark_3d_filter;
                    for (int j = 0; j < ransac_filter.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[already_triangulate_id[ransac_filter[j]]];


                        landmark_3d_filter.push_back(mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->point_);
                        cur_points_tracked_filter.push_back(cur_points[already_triangulate_id[ransac_filter[j]]]);
                        tracked_filter_id.push_back(already_triangulate_id[ransac_filter[j]]);

                        mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->triangulate_ = true;
                        tracked_filter_landmark.push_back(mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]);


                    }


                    // Refine Pose
                    solvePnP(landmark_3d_filter, cur_points_tracked_filter, K, cv::Mat(), cur_r_vector, cur_t, true, cv::SOLVEPNP_ITERATIVE);


                    cv::Rodrigues(cur_r_vector, cur_r);
                    std::cout << cur_t.at<double>(0, 0) << " " << cur_t.at<double>(1, 0) << " " << cur_t.at<double>(2, 0) << std::endl;

                }

                // Add key Frame
                mono_vo_map.key_frames_id_.push_back(i);
                mono_vo_map.key_frames_poses_.insert({ i,Pose(K, cur_r.clone(), cur_t.clone()) });

                // Update Landmark
                // tracked_filter_id is the triangulate filter
                // Do Not Need Triangulate
                for (int j = 0; j < tracked_filter_id.size(); j++)
                {
                    int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[tracked_filter_id[j]];

                    LandMark *landmark = mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id];
                    landmark->view_id_.push_back(i);
                    landmark->view_pt_.push_back(cur_points[tracked_filter_id[j]]);
                }
                // Needed Triangulate
                for (int j = 0; j < needed_triangulate_id.size(); j++)
                {
                    int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[needed_triangulate_id[j]];

                    LandMark *landmark = mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id];
                    landmark->view_id_.push_back(i);
                    landmark->view_pt_.push_back(cur_points[needed_triangulate_id[j]]);
                }

                // Triangulate Two Views
                cv::Mat P0 = mono_vo_map.key_frames_poses_[mono_vo_map.last_key_frame_id_].GetP();
                cv::Mat P1 = mono_vo_map.key_frames_poses_[i].GetP(); // i is current key frame id
                std::vector<cv::Point2f> point0, point1;


                std::vector<LandMark*> needed_triangulated_landmark;
                for (int j = 0; j < needed_triangulate_id.size(); j++)
                {
                    int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[needed_triangulate_id[j]];

                    LandMark *landmark = mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id];

                    // Need Triangulate
                    if (!landmark->triangulate_ && landmark->view_pt_.size() == 2)
                    {
                        auto landmark_view_pt_iter = landmark->view_pt_.begin();

                        point0.push_back(*landmark_view_pt_iter);


                        landmark_view_pt_iter++;
                        point1.push_back(*landmark_view_pt_iter);

                        needed_triangulated_landmark.push_back(landmark);
                    }
                }


                // The new Triangulate landmark
                if (needed_triangulated_landmark.size() > 0)
                {

                    cv::Mat triangulate_4d_results;
                    triangulatePoints(P0, P1, point0, point1, triangulate_4d_results);

                    for (int j = 0; j < needed_triangulated_landmark.size(); j++)
                    {
                        LandMark *landmark = needed_triangulated_landmark[j];
                        landmark->triangulate_ = true;
                        cv::Mat triangulate_4d_results_col = triangulate_4d_results.col(j);

                        landmark->point_.x = triangulate_4d_results_col.at<float>(0, 0) / triangulate_4d_results_col.at<float>(3, 0);
                        landmark->point_.y = triangulate_4d_results_col.at<float>(1, 0) / triangulate_4d_results_col.at<float>(3, 0);
                        landmark->point_.z = triangulate_4d_results_col.at<float>(2, 0) / triangulate_4d_results_col.at<float>(3, 0);

                    }


                    // Update Map
                    // Delete UnTracked Points
                    for (auto landmark_iter = mono_vo_map.landmarks_.begin(); landmark_iter != mono_vo_map.landmarks_.end(); landmark_iter++)
                    {
                        if ((*landmark_iter)->triangulate_ == false)
                        {
                            delete (*landmark_iter);
                            mono_vo_map.landmarks_.erase(landmark_iter);
                        }
                    }

                }

                // *****************************************************************
                // Add New Points
                // *****************************************************************

                // Detect New Points
                cv::Mat new_mask = mask.clone();
                for (int j = 0; j < tracked_filter_id.size(); j++)
                {

                    cv::Point2f &point = cur_points[tracked_filter_id[j]];
                    if (point.x > border && point.x<(image_width - border) && point.y>border && point.y < (image_height - border))
                    {
                        cv::circle(new_mask, point, POINT_DISTANCE, 0, -1);
                    }
                }
                for (int j = 0; j < needed_triangulate_id.size(); j++)
                {

                    cv::Point2f &point = cur_points[needed_triangulate_id[j]];
                    if (point.x > border && point.x<(image_width - border) && point.y>border && point.y < (image_height - border))
                    {
                        cv::circle(new_mask, point, POINT_DISTANCE, 0, -1);
                    }
                }


                std::vector<cv::Point2f> new_points;
                int add_new_points_number = MAX_POINT - tracked_filter_id.size() - needed_triangulate_id.size();
                if (add_new_points_number > 0)
                {
                    goodFeaturesToTrack(cur_frame_gray, new_points, add_new_points_number, 0.01, POINT_DISTANCE, new_mask);
                }
                if (new_points.size() > 0)
                    cornerSubPix(cur_frame_gray, new_points, subPixWinSize, cv::Size(-1, -1), termcrit);



                // Update Last Key Frame
                mono_vo_map.last_key_frame_id_ = i;
                mono_vo_map.last_key_frame_ = cur_frame.clone();
                mono_vo_map.last_key_frame_points.clear();
                mono_vo_map.last_key_frame_landmarks_.clear();
                for (int j = 0; j < tracked_filter_id.size(); j++)
                {
                    mono_vo_map.last_key_frame_landmarks_.push_back(tracked_filter_landmark[j]);
                    mono_vo_map.last_key_frame_points.push_back(cur_points[tracked_filter_id[j]]);
                }

                for (int j = 0; j < needed_triangulate_id.size(); j++)
                {
                    mono_vo_map.last_key_frame_landmarks_.push_back(needed_triangulated_landmark[j]);
                    mono_vo_map.last_key_frame_points.push_back(cur_points[needed_triangulate_id[j]]);
                }
                for (int j = 0; j < new_points.size(); j++)
                {
                    LandMark *landmark = new LandMark();
                    landmark->triangulate_ = false;
                    landmark->view_id_.push_back(i);
                    landmark->view_pt_.push_back(new_points[j]);

                    // Add to Last Key Frame
                    mono_vo_map.last_key_frame_landmarks_.push_back(landmark);
                    mono_vo_map.last_key_frame_points.push_back(new_points[j]);

                    // Add Landmarks
                    mono_vo_map.landmarks_.push_back(landmark);
                }

                // Update Last Frame
                mono_vo_map.last_frame_ = cur_frame.clone();
                mono_vo_map.last_frame_points.clear();
                mono_vo_map.last_frame_pyramid.clear();
                mono_vo_map.last_frame_tracked_last_key_frame_id_.clear();

                mono_vo_map.last_frame_pyramid.assign(cur_frame_pyramid.begin(), cur_frame_pyramid.end());
                for (int j = 0; j < mono_vo_map.last_key_frame_points.size(); j++)
                {
                    mono_vo_map.last_frame_points.push_back(mono_vo_map.last_key_frame_points[j]);
                    mono_vo_map.last_frame_tracked_last_key_frame_id_.push_back(j);
                }

            }
            else
            {

                // Not Init
                if (!mono_vo_map.INIT_)
                {
                    // Match By L-K
                    std::vector<cv::Point2f> cur_points_tracked;
                    std::vector<int> tracked_id;

                    // Get Match After L-K
                    for (int j = 0; j < track_status.size(); j++)
                    {
                        if (track_status[j])
                        {
                            cur_points_tracked.push_back(cur_points[j]);
                            tracked_id.push_back(mono_vo_map.last_frame_tracked_last_key_frame_id_[j]);
                        }
                    }


                    // Update last frame
                    mono_vo_map.last_frame_ = cur_frame.clone();
                    mono_vo_map.last_frame_pyramid.clear();
                    mono_vo_map.last_frame_points.clear();
                    mono_vo_map.last_frame_tracked_last_key_frame_id_.clear();

                    mono_vo_map.last_frame_pyramid.assign(cur_frame_pyramid.begin(), cur_frame_pyramid.end());

                    for (int j = 0; j < tracked_id.size(); j++)
                    {
                        mono_vo_map.last_frame_points.push_back(cur_points_tracked[j]);
                        mono_vo_map.last_frame_tracked_last_key_frame_id_.push_back(tracked_id[j]);
                    }

                    std::cout << "Not Init..." << std::endl;
                }
                else
                {
                    // Match By L-K
                    std::vector<cv::Point2f> last_key_frame_points_tracked, cur_points_tracked;
                    std::vector<int> tracked_id;

                    // Already Triangulate Points
                    std::vector<cv::Point2f> cur_points_tracked_already_triangulate;
                    std::vector<int> already_triangulate_id;

                    // Match After RANSAC
                    std::vector<cv::Point2f> last_key_frame_points_tracked_filter, cur_points_tracked_filter;
                    std::vector<int> tracked_filter_id;

                    // Get Match After L-K
                    for (int j = 0; j < track_status.size(); j++)
                    {
                        if (track_status[j])
                        {
                            int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[j];
                            last_key_frame_points_tracked.push_back(mono_vo_map.last_key_frame_points[mono_vo_map.last_frame_tracked_last_key_frame_id_[j]]);
                            cur_points_tracked.push_back(cur_points[j]);
                            tracked_id.push_back(j);
                        }
                    }

                    // Get the Tracked which has triangulated
                    for (int j = 0; j < tracked_id.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[tracked_id[j]];
                        if (mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->triangulate_)
                        {
                            cur_points_tracked_already_triangulate.push_back(cur_points[tracked_id[j]]);
                            already_triangulate_id.push_back(tracked_id[j]);
                        }
                    }


                    // Get Landmark By Track
                    std::vector<cv::Point3d> landmark_3d;
                    for (int j = 0; j < already_triangulate_id.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[already_triangulate_id[j]];

                        landmark_3d.push_back(mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->point_);

                    }
                    //std::cout << "Init Used landmark: " << already_triangulate_id.size() << std::endl;


                    // RANSAC With SolvePnP
                    cv::Rodrigues(cur_r, cur_r_vector);
                    vector<int> ransac_filter;
                    solvePnPRansac(landmark_3d, cur_points_tracked_already_triangulate, K, cv::Mat(), cur_r_vector, cur_t, false, 100, 8, 0.9899, ransac_filter, cv::SOLVEPNP_EPNP);


                    std::vector<cv::Point3d> landmark_3d_filter;
                    for (int j = 0; j < ransac_filter.size(); j++)
                    {
                        int last_key_frame_point_id = mono_vo_map.last_frame_tracked_last_key_frame_id_[already_triangulate_id[ransac_filter[j]]];


                        landmark_3d_filter.push_back(mono_vo_map.last_key_frame_landmarks_[last_key_frame_point_id]->point_);
                        cur_points_tracked_filter.push_back(cur_points[already_triangulate_id[ransac_filter[j]]]);
                        tracked_filter_id.push_back(already_triangulate_id[ransac_filter[j]]);


                    }


                    // Refine Pose
                    solvePnP(landmark_3d_filter, cur_points_tracked_filter, K, cv::Mat(), cur_r_vector, cur_t, true, cv::SOLVEPNP_ITERATIVE);


                    cv::Rodrigues(cur_r_vector, cur_r);
                    std::cout << cur_t.at<double>(0, 0) << " " << cur_t.at<double>(1, 0) << " " << cur_t.at<double>(2, 0) << std::endl;


                    // Update Last Frame

                    std::vector<int> id_temp;
                    id_temp.assign(mono_vo_map.last_frame_tracked_last_key_frame_id_.begin(), mono_vo_map.last_frame_tracked_last_key_frame_id_.end());

                    mono_vo_map.last_frame_ = cur_frame.clone();
                    mono_vo_map.last_frame_pyramid.clear();
                    mono_vo_map.last_frame_points.clear();
                    mono_vo_map.last_frame_tracked_last_key_frame_id_.clear();


                    mono_vo_map.last_frame_pyramid.assign(cur_frame_pyramid.begin(), cur_frame_pyramid.end());

                    for (int j = 0; j < tracked_id.size(); j++)
                    {
                        mono_vo_map.last_frame_points.push_back(cur_points[tracked_id[j]]);

                        int last_key_frame_point_id = id_temp[tracked_id[j]];
                        mono_vo_map.last_frame_tracked_last_key_frame_id_.push_back(last_key_frame_point_id);
                    }


                }

            }

            // Show Track
            for (int j = 0; j < cur_points.size(); j++)
            {
                cv::circle(cur_frame_gray, cur_points[j], 10, cv::Scalar::all(255), 3);
            }

            cv::imshow("Video", cur_frame_gray);
            cv::waitKey(30);
        }
    }





    return 0;

}
