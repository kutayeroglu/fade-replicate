def get_cleaned_results(results, coco_gt):
    '''
    Sanity check procedure.
    If cleaned are not satisfactory, problem is likely
      not with the data format.
    '''
    cleaned_results = [
    {
        "image_id": r["image_id"],
        "category_id": r["category_id"],
        "bbox": r["bbox"],
        "score": r["score"]
    }
    for r in results
    ]

    print("Cleaned Results:", cleaned_results[:5])

    coco_dt = coco_gt.loadRes(cleaned_results)
    coco_evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    confidence_threshold = 0.5
    filtered_results = [
        r for r in cleaned_results if r['score'] > confidence_threshold
    ]

    print("Filtered Results Count:", len(filtered_results))