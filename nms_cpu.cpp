// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include "cpu/vision.h"
#include <cmath>
#include <vector>

using std::vector;

template <typename scalar_t>
at::Tensor nms_cpu_kernel(at::Tensor& dets,             //boxes: [N, (x1, y1, x2, y2)].注意(y2, x2)可以超过box的边界.
                          const at::Tensor& scores,     //scores: box的分数的一维数组.
                          const float threshold) {      //threshold: float型. 用于过滤IoU的阈值.
  AT_ASSERTM(!dets.type().is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.type().is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(dets.type() == scores.type(), "dets should have the same type as scores");

  vector<int64_t> attach;

  const auto threshold_2 = 0.6;     //控制attach框与major框的关联紧密程度，此阈值越高则attach框越少

  if (dets.numel() == 0) {
    return at::empty({0}, dets.options().dtype(at::kLong).device(at::kCPU));
  }

  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();

  at::Tensor areas_t = (x2_t - x1_t + 1) * (y2_t - y1_t + 1);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t = at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data<uint8_t>();
  auto order = order_t.data<int64_t>();
  auto x1 = x1_t.data<scalar_t>();
  auto y1 = y1_t.data<scalar_t>();
  auto x2 = x2_t.data<scalar_t>();
  auto y2 = y2_t.data<scalar_t>();
  auto areas = areas_t.data<scalar_t>();

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<scalar_t>(0), xx2 - xx1 + 1);
      auto h = std::max(static_cast<scalar_t>(0), yy2 - yy1 + 1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= threshold){
        if(ovr * j >= threshold_2){     //关联判断阈值，越接近1，attach框越少
          attach.push_back(j);  //通过则加入attach
        }
        suppressed[j] = 1;
      }
    }
    attach.push_back(i);    //attach的最后加入主框，待会记得释放attach

    auto min_x1 = x1[attach[0]], max_y1 = y1[attach[0]], max_x2 = x2[attach[0]], min_y2 = y2[attach[0]];    //初始化四点为第一个框的坐标
    printf("min_x1:%g\tmax_y1:%g\tmax_x2:%g\tmin_y2:%g\n",min_x1,max_y1,max_x2,min_y2);

    for(int64_t _m = 1; _m <= attach.size(); _m++){
        auto m = attach[_m];
        auto ix1 = x1[m];
        auto iy1 = y1[m];
        auto ix2 = x2[m];
        auto iy2 = y2[m];
        if(ix1 < min_x1) min_x1 = ix1;  //找最小x1
        if(iy1 > max_y1) max_y1 = iy1;  //找最大y1
        if(ix2 > max_x2) max_x2 = ix2;  //找最大x2
        if(iy2 < min_y2) min_y2 = iy2;  //找最小y2
    }

    x1[i] = min_x1;
    y1[i] = max_y1;
    x2[i] = max_x2;
    y2[i] = min_y2;
    //更新det两点坐标

    attach.clear();     //释放attach临时空间，为下一个主框的扩张做准备; clear()函数清空vector

  }

  return at::nonzero(suppressed_t == 0).squeeze(1);
}

at::Tensor nms_cpu(at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {
  at::Tensor result;
  AT_DISPATCH_FLOATING_TYPES(dets.type(), "nms", [&] {
    result = nms_cpu_kernel<scalar_t>(dets, scores, threshold);
  });
  return result;
}
