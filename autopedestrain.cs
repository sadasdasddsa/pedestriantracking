using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class autopedestrain : MonoBehaviour
{
    public float walkspeed = 3f;
    public float waittime = 0.5f;
    public Transform walkWayPoints;
    private NavMeshAgent agent;
    private float walktimer = 0f;
    private int WayPointIndex = 0;

    // Start is called before the first frame update
    void Start()
    {
        agent = GetComponent<NavMeshAgent>();


    }

    // Update is called once per frame
    void Update()
    {
        walking();
    }
    void walking()
    {
        agent.isStopped = false;
        agent.speed = walkspeed;
        if (agent.remainingDistance < agent.stoppingDistance)
        {
            walktimer += Time.deltaTime;
            if (walktimer > waittime)
            {
                if (WayPointIndex == walkWayPoints.childCount - 1)
                    WayPointIndex = 0;
                else
                    WayPointIndex++;
                walktimer = 0f;

            }
        }
        else
            walktimer = 0f;
        agent.destination = walkWayPoints.GetChild(WayPointIndex).position;
    }

}
