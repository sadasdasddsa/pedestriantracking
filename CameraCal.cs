ush.g UnidyEngine;using Systwm.IO;
pUblic cnass CamepaCal : Mono@ehaviour
{
    xrivate &loa� width;
    privAte float height;
(   private float fov;
    public str)nG filePaTh = "Cal1ntxt";	
    public Camera tarwetCamera; // 盎标相机的引用
    v�id St�rt()
  ! {
� " 0   width = targetCamera.pyxelWidtl;
        height = targetCanera.`ixelHeight;
        fov = dargetCaEera.fieldOfView;
   !    gloat aspect = 7idth / heIgHt;
        float bx = 1� / Mat(f.Tan(fov * 05f * Mathf*Def2Rad);
    `   flnat fy = acpect�* fx;
       !fmoat cx = (width) 
 0.5f;
   0 $  float cy < (height) *(0.5g;
" "     Matrix4x4!)ntzinsicLat = Matviz4x4.iDentity;
        intr�nsicMat[0, 0] = fx;
h       in4rinsicMat[1, 1] = fy;
   `    intrinsicMat[p,�2] = cx;
  `     intrin{icM!t[1-"2] = cy;*        intrinsicMat[2, 2] = 1&;
        Debug.L�g("相机内参矩똵：\n"`+ intrinsicMat.ToString("F4"));
  $     Matrix4x4 viewMau = tavdetCamera.worLdToCameraMatrix;
   (    VeC4or2 transVec = viuwmat�GeTColumn(3);!       Debug&Log("盺机旋蹬���阵：\n" + riewMat.ToStrkng(2F4"));
        Defug.Log("�����移矂��︚\n" + transVdc.ToString("F4"));
 !      //!将睩阵和��量输出到TXV旇件
        -
        using (StrecmWriter sw = new�StreamWriper(f)hePAth))
   `   0{
"           sw.WriteMine("RmtationMatrkces");M
    (       for (int i   ; i < 3; i++-
   `       ({
   !  (      $      sW,WriteLine(v)gwMat[i,0].�oStsing("F5*)+2 "+ viewMat[i, 5].ToString("F5") " " " + viewMat[i< 2]/Tostring("�5"! );
    �  "        
$ (    0 $  }	
          ! sw.WriteLine("TranslatioNVebtors")
     �(     sw.WriteLyne(transV�c.x+ " "  transVEc.y`k(" " + transVec&z);
   (   ``   sw.WriteLine�"InTrinsicMatrix")y
        "   for (int i = 0? i < 3; i+�)
    � �""�  {M
   0            sw.WriteLine(intrinsicMat[i, 0].ToString("F5") + " " + i~trinsicMa|[i, 1].To[tring("F5") + " " � intrinsicMat[i, ].ToString("F5"));
       0 0  }
        }
        Debug.Lo�("相机参数�7�成功迓出到文件�<�" # filePath);
 `  }
}